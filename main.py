#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_curve, f1_score, recall_score, precision_score, matthews_corrcoef, accuracy_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=str, default='TNBC')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--model', type=str, default='LR')
    parser.add_argument('--average_method', type=str, default='binary')
    parser.add_argument('--n_folds', type=float, default=5)
    return parser.parse_args()


def init_model(args, C=1.0, model='LR'):
    if model == 'LR':
        model = LogisticRegression(penalty='l1',
                                   C=C,
                                   fit_intercept=True,
                                   intercept_scaling=1,
                                   class_weight='balanced',
                                   random_state=args.seed,
                                   solver='saga',
                                   max_iter=50,
                                   multi_class='auto',
                                   verbose=0,
                                   l1_ratio=None
                                   )
    return model

def read_data(args, cohort):
    
    features = pd.read_csv(os.path.join('features/IMPRESS/%s.csv' % cohort),
                           index_col=0, header=[0, 1])
    # drop constant features
    features.drop(('all','HE_proportion'),axis='columns', inplace=True)
    features.drop(('all','CD8_proportion'),axis='columns', inplace=True)
    features.drop(('all','CD163_proportion'),axis='columns', inplace=True)
    features.drop(('all','PDL1_proportion'),axis='columns', inplace=True)
    features.sort_index(inplace=True)
    
    clinical = pd.read_csv(os.path.join('clinical/%s.csv' % cohort),
                       index_col=0, header=[0, 1])
    y = clinical[('clinical', 'pCR')]

    if cohort == 'HER2+':
        X = pd.concat((features,
                       clinical.loc[:,[('clinical', 'age'),
                                       ('clinical', 'HER2/CEP17 ratio'),
                                       ('clinical', 'ER'),
                                       ('clinical', 'ER%'),
                                       ('clinical', 'PR'),
                                       ('clinical', 'PR%')]
                                    ]
                       ), axis=1)
    elif cohort == 'TNBC':
        X = pd.concat((features,
                       clinical.loc[:,[('clinical', 'age')]
                                    ]
                       ), axis=1)
    
    p_eval = pd.read_csv(os.path.join('features/pathologists/%s.csv' % cohort),
                       index_col=0, header=[0, 1])
    
    return X, y, p_eval, clinical

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
    
def leave_one_out_cv_dict(args, X, y):
    y = y.values
    pids = X.index.values
    
    dataset_split = {}
    scaler_list = []
    for test_idx in range(len(y)):
        tset = 'testing %02d' % test_idx
        dataset_split[tset] = {}
        dataset_split[tset]['train_val'] = {}
        dataset_split[tset]['hold_out_testing'] = {}
        testing_index = [test_idx]
        train_val_index = [i for i in range(len(y)) if i not in testing_index]
    
        dataset_split[tset]['hold_out_testing']['pid'] = pids[testing_index]
        dataset_split[tset]['hold_out_testing']['X'] = X.iloc[testing_index,:]
        dataset_split[tset]['hold_out_testing']['X_original'] = X.iloc[testing_index,:]
        dataset_split[tset]['hold_out_testing']['y'] = y[testing_index]
        
        dataset_split[tset]['train_val']['pid'] = pids[train_val_index]
        dataset_split[tset]['train_val']['X'] = X.iloc[train_val_index,:]
        dataset_split[tset]['train_val']['X_original'] = X.iloc[train_val_index,:]
        dataset_split[tset]['train_val']['y'] = y[train_val_index]
        
        # standardize
        scaler = StandardScaler()
        scaler.fit(dataset_split[tset]['train_val']['X'])
        scaler_list.append(scaler)
        dataset_split[tset]['train_val']['X'] = scaler.transform(dataset_split[tset]['train_val']['X'])
        dataset_split[tset]['hold_out_testing']['X'] = scaler.transform(dataset_split[tset]['hold_out_testing']['X'])
        
        dataset_split[tset]['train_val']['X'] = pd.DataFrame(dataset_split[tset]['train_val']['X'],
                                                       index = dataset_split[tset]['train_val']['X_original'].index,
                                                       columns = dataset_split[tset]['train_val']['X_original'].columns)
        dataset_split[tset]['hold_out_testing']['X'] = pd.DataFrame(dataset_split[tset]['hold_out_testing']['X'],
                                                       index = dataset_split[tset]['hold_out_testing']['X_original'].index,
                                                       columns = dataset_split[tset]['hold_out_testing']['X_original'].columns)
        
        kf = KFold(n_splits=args.n_folds, random_state=args.seed, shuffle=True)    
        
        for i, (train_index, val_index) in enumerate(kf.split(dataset_split[tset]['train_val']['pid'])):
            dataset_split[tset]['train_val']['fold ' + str(i+1)] = {}
            dataset_split[tset]['train_val']['fold ' + str(i+1)]['train'] = {}
            dataset_split[tset]['train_val']['fold ' + str(i+1)]['train']['pid'] = dataset_split[tset]['train_val']['pid'][train_index]
            dataset_split[tset]['train_val']['fold ' + str(i+1)]['train']['X'] = dataset_split[tset]['train_val']['X'].iloc[train_index,:]
            dataset_split[tset]['train_val']['fold ' + str(i+1)]['train']['X_original'] = dataset_split[tset]['train_val']['X_original'].iloc[train_index,:]
            dataset_split[tset]['train_val']['fold ' + str(i+1)]['train']['y'] = dataset_split[tset]['train_val']['y'][train_index]
    
            dataset_split[tset]['train_val']['fold ' + str(i+1)]['val'] = {}
            dataset_split[tset]['train_val']['fold ' + str(i+1)]['val']['pid'] = dataset_split[tset]['train_val']['pid'][val_index]
            dataset_split[tset]['train_val']['fold ' + str(i+1)]['val']['X'] = dataset_split[tset]['train_val']['X'].iloc[val_index,:]
            dataset_split[tset]['train_val']['fold ' + str(i+1)]['val']['X_original'] = dataset_split[tset]['train_val']['X_original'].iloc[val_index,:]
            dataset_split[tset]['train_val']['fold ' + str(i+1)]['val']['y'] = dataset_split[tset]['train_val']['y'][val_index]
        
    return dataset_split, scaler_list, kf

    
def run_cross_validation(args, X, dataset_split, hyperparameters, verbose=0):
    # Finding optimal hyperparameter
    hparam_df = pd.DataFrame(index = ['fold %d' % (f+1) for f in np.arange(args.n_folds)],
                         columns = hyperparameters)
    for i, C in enumerate(hyperparameters):
        model = init_model(args, C=C, model=args.model)
        if verbose: print("[%d/%d] current penalty: %.4E" %((i+1), len(hyperparameters), C))
        for i in range(args.n_folds):
            fold = i+1
            if verbose: print("Cross-validation -- %d/%d" % (fold,args.n_folds))
            datasets = {'train':{}, 'val':{}}
            datasets['train'] = {'X':dataset_split['train_val']['fold %d' % fold]['train']['X'],
                                'y':dataset_split['train_val']['fold %d' % fold]['train']['y']}
            datasets['val'] = {'X':dataset_split['train_val']['fold %d' % fold]['val']['X'],
                                'y':dataset_split['train_val']['fold %d' % fold]['val']['y']}
            clf = model.fit(datasets['train']['X'], datasets['train']['y'])
            y_true = datasets['val']['y']
            y_pred = clf.predict(datasets['val']['X'])
            y_proba = clf.predict_proba(datasets['val']['X'])[:,1]
            performance = get_evaluation_res(y_true, y_pred, y_proba, args.average_method)
            hparam_df.loc['fold %d' % fold, C] = performance['AUC']
            
    optimal_C = hparam_df.mean(axis=0).idxmax()
    if verbose: print("Optimal penalty: %.4E, optimal Mean AUC on validation: %.10f" % (optimal_C, np.max(hparam_df.mean(axis=0))))
    hparam_df.to_csv(os.path.join(writedir, 'LR_validation_performances.csv'))
    # Final model
    model = init_model(args, C=optimal_C, model=args.model) 
    datasets = {'train':{}, 'test':{}}
    datasets['train'] = {'X':dataset_split['train_val']['X'],
                        'y':dataset_split['train_val']['y']}
    datasets['test'] = {'X':dataset_split['hold_out_testing']['X'],
                        'y':dataset_split['hold_out_testing']['y']}
    clf = model.fit(datasets['train']['X'], datasets['train']['y'])
    y_true = datasets['test']['y']
    y_pred = clf.predict(datasets['test']['X'])
    y_proba = clf.predict_proba(datasets['test']['X'])[:,1]
    performance = get_evaluation_res(y_true, y_pred, y_proba, args.average_method)
    
    #  If it is negative, it would be a decrease in residual tumor probability.
    coefficients = pd.DataFrame(index = X.columns, columns=['coefficient'])
    if args.model == 'LR':
        coefficients['coefficient'] = clf.coef_.reshape(-1)
    return clf, optimal_C, hparam_df, performance, y_true, y_pred, y_proba, coefficients

def run_leave_one_out(args, X, dataset_split, hyperparameters, verbose=0):
    y_true_all = np.array([])
    y_pred_all = np.array([])
    y_proba_all = np.array([])
    optimal_C_all = []
    model_list = []
    coefficients = None
    print('Running leave one out ...')
    for loo in tqdm(dataset_split.keys()):
        dataset_split_curr = dataset_split[loo]
        clf, optimal_C, hparam_df, performance, y_true, y_pred, y_proba, coef = \
            run_cross_validation(args, X, dataset_split_curr, hyperparameters, verbose)
        y_true_all = np.concatenate((y_true_all, y_true)).astype(int)
        y_pred_all = np.concatenate((y_pred_all, y_pred)).astype(int)
        y_proba_all = np.concatenate((y_proba_all, y_proba))
        optimal_C_all.append(optimal_C)
        model_list.append(clf)
        if coefficients is None:
            coefficients = coef
        else:
            coefficients = pd.concat((coefficients, coef),axis=1)
    performance = get_evaluation_res(y_true_all, y_pred_all, y_proba_all, args.average_method)
    return model_list, optimal_C_all, performance, y_true_all, y_pred_all, y_proba_all, coefficients


if __name__=='__main__':
    args = parse_args()
    args.model='LR'
    random.seed(args.seed)
    np.random.seed(args.seed)
    plt.ioff()
    cohort = args.cohort
    print('Cohort: %s    Seed: %d' % (cohort, args.seed))
    writedir_root = os.path.join('results', cohort)
    X_in, y, p_eval, clinical = read_data(args, cohort)
    
    
    for fgroup in ['IMPRESS_all', 'IMPRESS_HE_only', 'IMPRESS_IHC_only', 'pathologists_eval']:
        writedir = os.path.join(writedir_root, fgroup, 'seed=%d' % args.seed)
        if fgroup == 'IMPRESS_all':
            X = copy.deepcopy(X_in)
        if fgroup == 'IMPRESS_HE_only':
            selected_features = [('stroma','HE_proportion'), ('tumor','HE_proportion'), ('lymphocytes','HE_proportion')]
            X = copy.deepcopy(pd.concat((X_in.loc[:, selected_features], X_in.loc[:, 'clinical' == X_in.columns.get_level_values(0)]), axis=1))
        if fgroup == 'IMPRESS_IHC_only':
            selected_features = [('all','CD8_purity'), ('all','CD163_purity'), ('all','PDL1_purity')]
            X = copy.deepcopy(pd.concat((X_in.loc[:, selected_features], X_in.loc[:, 'clinical' == X_in.columns.get_level_values(0)]), axis=1))
        if fgroup == 'pathologists_eval':
            X = copy.deepcopy(X_in)
            X = pd.concat((p_eval.loc[X.index,:], X.loc[:, 'clinical' == X.columns.get_level_values(0)]), axis=1)
        
        if not os.path.exists(writedir): os.makedirs(writedir)
        
        dataset_split, scaler_list, kf = leave_one_out_cv_dict(args, X, y)
        with open(os.path.join(writedir, 'dataset_split.pkl'),'wb') as f: pickle.dump(dataset_split, f)
        
    # =============================================================================
    #     Logistic regression model setup
    # =============================================================================
        hyperparameters = np.arange(1,11)*0.1
        with open(os.path.join(writedir,'hyperparameter_search_list.txt'),'w+') as f:
            f.write(str(list(hyperparameters)))
            
        model_list, optimal_C_all, performance, y_true, y_pred, y_proba, coefficients = \
            run_leave_one_out(args, X, dataset_split, hyperparameters, verbose=0)
            
        # print(performance)
        # print('\n\nAUC =',performance['AUC'],'\n')
        
        performance_df = pd.DataFrame(performance, index=['performance'])
        performance_df.to_csv(os.path.join(writedir, 'performance.csv'))
        coefficients.to_csv(os.path.join(writedir, 'coefficients.csv'))
        y_all = pd.DataFrame(np.stack((y_true,y_pred,y_proba),axis=1), columns=['y_true','y_pred','y_pred_proba'], index = X.index)
        y_all.to_csv(os.path.join(writedir, 'y_all.csv'))
        coefficient = coefficients.mean(axis=1)
    
    
    
    
    
