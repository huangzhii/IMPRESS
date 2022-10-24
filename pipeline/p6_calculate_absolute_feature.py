#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 09:44:01 2022

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
import multiprocessing as mp


opj = os.path.join
workdir = 'workdir'
from python_script import get_data
from python_script import plots




def parfun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

def parmap(f, X, nprocs=mp.cpu_count()):
    q_in = mp.Queue(1)
    q_out = mp.Queue()
    proc = [mp.Process(target=parfun, args=(f, q_in, q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()
    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]
    [p.join() for p in proc]
    return [x for i, x in sorted(res)]


def read_he_ihc_seg(resultdir, dlepoch, k):
    HE_seg = Image.open(os.path.join(resultdir,'Full_resolution_HE_seg_grey_dlepoch=%d.png' % dlepoch))
    IHC_seg = Image.open(os.path.join(resultdir,'Full_resolution_IHC_nr_kmeans_seg_k=%d.png' % k))
    HE_seg = np.array(HE_seg)
    IHC_seg = np.array(IHC_seg)
    
    if HE_seg.shape != IHC_seg.shape:
        if HE_seg.shape[0] <= IHC_seg.shape[0] and HE_seg.shape[1] <= IHC_seg.shape[1]:
            HE_seg2 = np.zeros(IHC_seg.shape)
            HE_seg2[0:HE_seg.shape[0], 0:HE_seg.shape[1]] = HE_seg
            HE_seg = HE_seg2
            del HE_seg2
        else:
            raise ValueError('HE segmentation size %s dismatch IHC segmentation size %s!' % (HE_seg.shape, IHC_seg.shape))
    return HE_seg, IHC_seg

def calculate_absolute_feature(HE_seg, IHC_seg):
    HE_seg_dict = {'exclude':0, 'stroma': 1, 'tumor': 2, 'lymphocytes':3}
    IHC_seg_dict = {'background':0, 'CD8': 1, 'CD163': 2, 'PDL1':3}
    
    feature_stat = pd.DataFrame(index=['IHC background pixel',
                                       'IHC CD8 pixel',
                                       'IHC CD163 pixel',
                                       'IHC PDL1 pixel'],
                                columns=['H&E exclude pixel',
                                       'H&E stroma pixel',
                                       'H&E tumor pixel',
                                       'H&E lymphocytes pixel'])
    
    for k_HE in HE_seg_dict:
        for k_IHC in IHC_seg_dict:
            pass
            pixels = np.sum((HE_seg == HE_seg_dict[k_HE]) & (IHC_seg == IHC_seg_dict[k_IHC]))
            feature_stat.loc['IHC %s pixel' % k_IHC, 'H&E %s pixel' % k_HE] = pixels
    return feature_stat

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=str, default='TNBC')
    parser.add_argument('--downsample', type=str, default='32x')
    parser.add_argument('--HE_seg_dlepoch', type=int)
    parser.add_argument('--k', type=int)
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    np.random.seed(0)
    plt.ioff()
    cohort = args.cohort
    dlepoch = args.HE_seg_dlepoch
    k = args.k
    
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
        
    
    def parallel_func(pid_tid):
        pbar.update(mp.cpu_count())
        resultdir = opj(workdir, 'results', cohort, '4_non_linear_results_%s_downsampled' % args.downsample, pid_tid)
        HE_seg, IHC_seg = read_he_ihc_seg(resultdir, dlepoch, k)
        feature_stat = calculate_absolute_feature(HE_seg, IHC_seg)
        feature_stat.to_csv(os.path.join(resultdir,'feature_stat.csv'))

    pbar = tqdm(total=int(len(tissue_df)))
        
    l = []
    for i in tqdm(tissue_df.index):
        pid, tid, path2HE, path2IHC = tissue_df.iloc[i,:]
        pid_tid = '%s_%s' % (pid, tid)
        l.append(pid_tid)
    parmap(lambda pid_tid: parallel_func(pid_tid), l)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
