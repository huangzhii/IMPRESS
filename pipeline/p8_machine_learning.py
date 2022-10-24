#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:18:44 2022

@author: zhihuang
"""


import os,sys,platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import copy
from tqdm import tqdm
import random
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_curve, f1_score, recall_score, precision_score, matthews_corrcoef, accuracy_score

opj = os.path.join
workdir = 'workdir'
from python_script import get_data
from python_script import plots


def read_data(args, workdir, datadir, cohort):
    features = pd.read_csv(opj(workdir, 'results', cohort, '7_extracted_features.csv'),
                           index_col=0, header=[0, 1])
    
    # drop useless features
    features.drop(('all','HE_proportion'),axis='columns', inplace=True)
    features.drop(('all','CD8_proportion'),axis='columns', inplace=True)
    features.drop(('all','CD163_proportion'),axis='columns', inplace=True)
    features.drop(('all','PDL1_proportion'),axis='columns', inplace=True)
    features.sort_index(inplace=True)
    X = copy.deepcopy(features)
    # X.columns = X.columns.to_flat_index()
    
    clinical = pd.read_excel(opj(workdir,'clinical_data','validation.xlsx'))
    clinical.dropna(subset=['age'], inplace=True)
    clinical['PCR'] = (clinical['PCR'] == 'Y').astype(int)
    
    if cohort == 'HER2+':
        clinical = clinical.loc[['HER2' in v for v in clinical['Label-ID']]]
    elif cohort == 'TNBC':
        clinical = clinical.loc[['TNBC' in v for v in clinical['Label-ID']]]
        
        
    clinical.index = [v.split('-')[1] for v in clinical['Label-ID']]
    clinical = clinical[['age','HER2 ratio','ER (pos-1, neg-0)','ER (%)','PR (Pos-1, neg-0)',\
                         'PR (%)', 'NG (1-3)', 'Nuclear grade (1-3)' , 'residual tumor size (cm)',\
                         'PCR']]
    
    feature_detail = ['age','HER2/CEP17 ratio', 'ER', 'ER%', 'PR', 'PR%', 'NG', 'Nuclear grade', 'Residual tumor size (cm)', 'pCR']
    clinical.columns = pd.MultiIndex.from_product([['clinical'],feature_detail], names=['region', 'feature'])
    
    mutual_pid = np.sort(np.intersect1d(X.index, clinical.index))
    print('Total available patients: %d' % len(mutual_pid))
    X = X.loc[mutual_pid,]
    clinical = clinical.loc[mutual_pid,]
    if args.use_clinical:
        if args.cohort == 'HER2+':
            X = pd.concat((X,
                           clinical.loc[:,[('clinical', 'age'),
                                           ('clinical', 'HER2/CEP17 ratio'),
                                           ('clinical', 'ER'),
                                           ('clinical', 'ER%'),
                                           ('clinical', 'PR'),
                                           ('clinical', 'PR%')]
                                        ]
                           ), axis=1)
        elif args.cohort == 'TNBC':
            X = pd.concat((X,
                           clinical.loc[:,[('clinical', 'age')]
                                        ]
                           ), axis=1)
    y = clinical[('clinical', 'pCR')]
        
    return X, y, clinical

def read_pathologists_evaluations(datadir, cohort, clinical):
    # read pathologists evaluations
    if cohort == 'HER2+':
        pathologist_eva = pd.read_excel(os.path.join(datadir[cohort],'..','PDL1-CD8-CD163 expression in HER2+ and TNBC.xlsx'), sheet_name=cohort)
        pathologist_eva.dropna(subset=['residual tumor (no-0, yes-1)'],inplace=True)
        pathologist_eva['pCR'] = 1-pathologist_eva['residual tumor (no-0, yes-1)'].astype(int)
        pathologist_eva.drop('residual tumor (no-0, yes-1)', axis='columns', inplace=True)
        pathologist_eva.set_index('unique ID',drop=True,inplace=True)
        pathologist_eva.sort_index(inplace=True)
        pathologist_eva = pathologist_eva.loc[clinical.index,:]
        # verify
        pCR_match = pathologist_eva['pCR'] == clinical[('clinical', 'pCR')]
        if pCR_match.sum() == len(pCR_match):
            print('verification passed.')
        pathologist_eva['PD-L1-tumor (cutoff 1%, 1- positive, 0-negative)'] = [int(str(v)[0]) for v in pathologist_eva['PD-L1-tumor (cutoff 1%, 1- positive, 0-negative)']]
        pathologist_eva.drop(['Case#','Case #','pCR'], axis='columns', inplace=True)
    if cohort == 'TNBC':
        pathologist_eva = pd.read_excel(os.path.join(datadir[cohort],'../..','PDL1-CD8-CD163 expression in HER2+ and TNBC.xlsx'), sheet_name=cohort)
        pathologist_eva['bx #'] = [bx.split(' ')[0] for bx in pathologist_eva['bx #']]
        case_match = pd.read_csv(os.path.join(datadir[cohort], '../UniqID-Case_match.csv'))
        matched_case_list = []
        for idx in pathologist_eva.index:
            bx = pathologist_eva.loc[idx,'bx #']
            if '-' in bx and len(bx.split('-')[1]) < 5:
                bx = '%s-%05d' % (bx.split('-')[0], int(bx.split('-')[1]))
                pathologist_eva.loc[idx,'bx #'] = bx
            matched_case = case_match.loc[case_match['Case'] == bx, :]
            if not len(matched_case):
                continue
            matched_case_list.append(bx)
            pid = matched_case['Unnamed: 1'].values[0]*100 + matched_case['Unnamed: 2'].values[0]
            pathologist_eva.loc[idx,'unique ID'] = pid
        
        pathologist_eva.dropna(subset=['unique ID'], inplace=True)
        pathologist_eva['unique ID'] = pathologist_eva['unique ID'].astype(int)
        pathologist_eva.sort_values(by='unique ID', inplace=True)
        pathologist_eva.set_index('unique ID', inplace=True)
        pathologist_eva = pathologist_eva.loc[clinical.index,:]
        pathologist_eva.drop(['bx #'], axis='columns', inplace=True)
        pathologist_eva = pathologist_eva.loc[pathologist_eva['PD1 total (%)']!='NT',:]
        
    iterables = [['pathologists'], pathologist_eva.columns]
    columns = pd.MultiIndex.from_product(iterables, names=['region', 'feature'])
    pathologist_eva.columns = columns
    return pathologist_eva


def get_evaluation_res(y_true, y_pred, y_pred_proba, average_method='macro'):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auroc = auc(fpr, tpr)
    f1 = f1_score(y_true, y_pred, average = average_method)
    precision = precision_score(y_true, y_pred, average = average_method)
    recall = recall_score(y_true, y_pred, average = average_method)
    mcc = matthews_corrcoef(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    tp,fp,tn,fn = 0,0,0,0
    for i in range(len(y_pred)): 
        if y_true[i]==y_pred[i]==1:
           tp += 1
        if y_pred[i]==1 and y_true[i]!=y_pred[i]:
           fp += 1
        if y_true[i]==y_pred[i]==0:
           tn += 1
        if y_pred[i]==0 and y_true[i]!=y_pred[i]:
           fn += 1
    if (tp+fn) == 0: sensitivity = np.nan
    else: sensitivity = tp/(tp+fn) # recall
    if (tn+fp) == 0: specificity = np.nan
    else: specificity = tn/(tn+fp)
    if (tp+fp) == 0: ppv = np.nan
    else: ppv = tp/(tp+fp) # precision or positive predictive value (PPV)
    if (tn+fn) == 0: npv = np.nan
    else: npv = tn/(tn+fn) # negative predictive value (NPV)
    if (tp+tn+fp+fn) == 0: hitrate = np.nan
    else: hitrate = (tp+tn)/(tp+tn+fp+fn) # accuracy (ACC)
    performance = {'Accuracy': acc,
                   'AUC': auroc,
                   'F-1': f1,
                   'precision': precision,
                   'recall': recall,
                   'mcc': mcc,
                   'tp': tp,
                   'fp': fp,
                   'tn': tn,
                   'fn': fn,
                   'sensitivity': sensitivity,
                   'specificity': specificity,
                   'ppv': ppv,
                   'npv': npv,
                   'hitrate': hitrate}
    return performance
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=str, default='HER2+')
    parser.add_argument('--downsample', type=str, default='32x')
    parser.add_argument('--model', type=str, default='LR')
    parser.add_argument('--average_method', type=str, default='binary')
    parser.add_argument('--leave_one_out', type=bool, default=True)
    parser.add_argument('--use_clinical', type=bool, default=True)
    parser.add_argument('--use_partial_feature', type=bool, default=False)
    parser.add_argument('--use_pathologists_evaluations', type=bool, default=False)
    return parser.parse_args()


if __name__=='__main__':
    args = parse_args()
    args.model='LR'
    # plt.ioff()
    cohort = args.cohort
    print('Cohort: %s' % cohort)
    writedir_root = os.path.join(opj(workdir, 'results', cohort, '5_machine_learning_results'))
    if args.leave_one_out: writedir_root = os.path.join(writedir_root, 'leave_one_out')
    else: writedir_root = os.path.join(writedir_root, 'hold_out_test=%.2f' % args.testing_split)
    
    datadir = {}
    datadir['HER2+'] = opj(workdir, '..', 'HER2+/')
    datadir['TNBC'] = opj(workdir, '..', 'TNBC/')
    
    tissuedir = {}
    tissuedir['HER2+'] = opj(workdir, 'results', 'HER2+', '3_tissues_%s_downsampled' % args.downsample, 'tissue')
    tissuedir['TNBC'] = opj(workdir, 'results', 'TNBC', '3_tissues_%s_downsampled' % args.downsample, 'tissue')
    
    landmarkdir = {}
    landmarkdir['HER2+'] = None
    landmarkdir['TNBC'] = None
    
    patient_df = get_data.get_patient_list(datadir, cohort=cohort)
    tissue_df = get_data.get_tissue_list(tissuedir, cohort=cohort)
    
    
    X_in, y, clinical = read_data(args, workdir, datadir, cohort)
    
    writedir_root = os.path.join(writedir_root, 'use_clinical=%s' % args.use_clinical)
    os.makedirs(writedir_root, exist_ok=True)
    
    groups = ['IMPRESS_all', 'IMPRESS_HE_only', 'IMPRESS_IHC_only']
    for fgroup in groups:
        if fgroup == 'IMPRESS_all':
            X = copy.deepcopy(X_in)
        if fgroup == 'IMPRESS_HE_only':
            selected_features = [('stroma','HE_proportion'), ('tumor','HE_proportion'), ('lymphocytes','HE_proportion')]
            X = copy.deepcopy(pd.concat((X_in.loc[:, selected_features], X_in.loc[:, 'clinical' == X_in.columns.get_level_values(0)]), axis=1))
        if fgroup == 'IMPRESS_IHC_only':
            selected_features = [('all','CD8_purity'), ('all','CD163_purity'), ('all','PDL1_purity')]
            X = copy.deepcopy(pd.concat((X_in.loc[:, selected_features], X_in.loc[:, 'clinical' == X_in.columns.get_level_values(0)]), axis=1))
            
        
        performace_df = pd.DataFrame()
        
        y_pred_proba_all_LOO_allseed = []
        for seed in range(1, 21):
            args.seed = seed
            random.seed(args.seed)
            np.random.seed(args.seed)
            # print('[group: %s; seed: %d]' % (fgroup, args.seed))
            
            # load model
            model_dir = opj(workdir, 'results', \
                            args.cohort, fgroup, 'seed=%d'%args.seed,'LR_model_leave_one_out')
            scaler_dir = opj(workdir, 'results', \
                            args.cohort, fgroup, 'seed=%d'%args.seed,'LR_standard_scaler_leave_one_out')
            y_pred_proba_all_LOO = []
            for model_file in os.listdir(model_dir):
                LOO_ID = model_file.split('_')[1]
                with open(opj(model_dir, model_file), 'rb') as f:
                    model = pickle.load(f)
                with open(opj(scaler_dir, 'scaler_%s' % LOO_ID), 'rb') as f:
                    scaler = pickle.load(f)
                X_scaled = scaler.transform(X)
                y_pred_proba = model.predict_proba(X_scaled)
                y_pred_proba_all_LOO.append(y_pred_proba)
            y_pred_proba_all_LOO = np.stack(y_pred_proba_all_LOO)
            y_pred_proba_all_LOO_allseed.append(y_pred_proba_all_LOO)
            y_pred_proba = np.mean(y_pred_proba_all_LOO, axis=0)[:,1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            performance = get_evaluation_res(y, y_pred, y_pred_proba, args.average_method)
            row = pd.DataFrame(performance,index=[seed])
            performace_df = pd.concat([performace_df, row])
    
        y_pred_proba_all_LOO_allseed = np.stack(y_pred_proba_all_LOO_allseed)
        
        performance_summary = pd.concat([performace_df.mean(axis=0), performace_df.std(axis=0)], axis=1)
        performance_summary.columns = ['mean','std']
        performance_summary.to_csv(opj(writedir_root, '%s_performance.csv' % fgroup))
    


