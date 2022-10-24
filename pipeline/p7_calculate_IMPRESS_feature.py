#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:12:21 2022

@author: zhihuang
"""



import os,sys,platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imshowpair import imshowpair
import pickle
import argparse
from tqdm import tqdm
from PIL import Image
import time
from sklearn.cluster import KMeans, MiniBatchKMeans
from skimage import color
import random
import glob
import re

opj = os.path.join
workdir = 'workdir'
from python_script import get_data
from python_script import plots


def get_feature_value(feature_stat, regionsum, cellsum_in, region, feature):
    if feature == 'HE_proportion':
        if region == 'all':
            value = 1
        else:
            value = (regionsum[[i for i in regionsum.index if region in i]] / regionsum[1:].sum()).values[0]
    else:
        cell, property = feature.split('_')
        if region == 'all':
            numerator = cellsum_in.loc[[cell in c for c in cellsum_in.index]]
            if property == 'ratio':
                denominator = regionsum.iloc[1:].sum()
            if property == 'proportion':
                denominator = numerator
            if property == 'purity':
                denominator = feature_stat.iloc[1:,1:].sum().sum()
            value = (numerator / denominator).values[0]
        else:
            numerator = feature_stat.loc[[cell in c for c in feature_stat.index],
                               [region in c for c in feature_stat.columns]]
            if property == 'ratio':
                denominator = regionsum.loc[[region in c for c in regionsum.index]]
            if property == 'proportion':
                denominator = cellsum_in.loc[[cell in c for c in cellsum_in.index]][0]
            if property == 'purity':
                denominator = feature_stat.loc[:,[region in c for c in regionsum.index]].iloc[1:].sum()
            value = (numerator / denominator).values[0][0]
    return value



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=str, default='HER2+')
    parser.add_argument('--downsample', type=str, default='32x')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    np.random.seed(0)
    plt.ioff()
    cohort = args.cohort
    
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
        
    
    
    feature_stat_pid = {}
    for i in tqdm(tissue_df.index):
        pid, tid, path2HE, path2IHC = tissue_df.iloc[i,:]
        pid_tid = '%s_%s' % (pid, tid)
        resultdir = opj(workdir, 'results', cohort, '4_non_linear_results_%s_downsampled' % args.downsample, pid_tid)
        if not os.path.exists(resultdir):
            continue
        feature_stat = pd.read_csv(os.path.join(resultdir,'feature_stat.csv'), index_col=0)
        if pid not in feature_stat_pid:
            feature_stat_pid[pid] = feature_stat
        else:
            feature_stat_pid[pid] += feature_stat
            
            
            
# =============================================================================
#     Construct features
# =============================================================================
    '''
    Proportion: (object pixels in H&E region pixels) / (H&E region pixels)
    Purity: (object pixels in H&E region pixels) / (object pixels)
    
    Example:
        (stroma, CD8_proportion): (CD8 pixels in stroma pixels)/(stroma pixels)
        (stroma, CD8_purity): (CD8 pixels in stroma pixels)/(CD8 pixels)
    '''
    
    HE_region = ['stroma', 'tumor', 'lymphocytes', 'all']
    feature_detail = ['HE_proportion',
                      'CD8_ratio','CD8_proportion','CD8_purity',
                      'CD163_ratio','CD163_proportion','CD163_purity',
                      'PDL1_ratio','PDL1_proportion','PDL1_purity']
    iterables = [HE_region,
                 feature_detail]
    columns = pd.MultiIndex.from_product(iterables, names=['region', 'feature'])
    
    pids = np.sort(list(feature_stat_pid.keys()))
    features = pd.DataFrame(index = pids, columns = columns)
    
    for pid in tqdm(pids):
        df = feature_stat_pid[pid]
        regionsum = df.sum(axis=0)
        cellsum_in = df.iloc[:,1:].sum(axis=1)
        for region in HE_region:
            for feature in feature_detail:
                # print(region,feature)
                value = get_feature_value(df, regionsum, cellsum_in, region, feature)
                    
                features.loc[pid, (region, feature)] = value
    
    
    features.to_csv(opj(workdir, 'results', cohort, '7_extracted_features.csv'))
            
            
            
