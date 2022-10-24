#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 17:07:57 2022

@author: zhihuang
"""


import os,sys,platform
import numpy as np
import pandas as pd
import openslide
from PIL import Image, ImageCms
import matplotlib.pyplot as plt
from imshowpair import imshowpair
import pickle
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import cv2
from sklearn.cluster import KMeans
from skimage import color

opj = os.path.join
workdir='workdir'


from python_script import get_data
from python_script import plots


def get_patch_summary(HE_seg,
                      patchsize):
    imgwidth, imgheight = HE_seg.size
    number_of_patch = 0
    summary = pd.DataFrame(columns=['i','j','bg_ratio'])
    for i in range(imgheight//patchsize):
        for j in range(imgwidth//patchsize):
            number_of_patch +=1
            box = (j*patchsize, i*patchsize, (j+1)*patchsize, (i+1)*patchsize)
            piece_seg = np.array(HE_seg.crop(box))
            bg_ratio = np.unique(piece_seg, return_counts=True)[1][0] / patchsize**2
            summary = summary.append({'i':i,
                                      'j':j,
                                      'bg_ratio':bg_ratio},
                                     ignore_index=True)
    summary.sort_values('bg_ratio', inplace=True)
    return summary

def top_patch(IHC_tissue_rgb,
              HE_seg,
              patchsize,
              top_m):
    
    summary = get_patch_summary(HE_seg, patchsize)
    summary.sort_values('bg_ratio', inplace=True)
    pv_lab = []
    pv_rgb = []
    for t in range(top_m):
        i, j, bg_ratio = summary.iloc[t]
        i, j = int(i), int(j)
        box = (j*patchsize, i*patchsize, (j+1)*patchsize, (i+1)*patchsize)
        piece_ihc_rgb = IHC_tissue_rgb[box[1]:box[3],box[0]:box[2],:]
        piece_seg = np.array(HE_seg.crop(box))
        pixel_values = piece_ihc_rgb#.reshape((-1, 3))
        # convert to float
        pixel_values = np.float32(pixel_values)
        pv_rgb.append(pixel_values)
    return pv_rgb, summary

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=str, default='TNBC', choices=['HER2+', 'TNBC'])
    parser.add_argument('--downsample', type=str, default='32x')
    parser.add_argument('--resolution', default='20x', type=str)
    return parser.parse_args()


if __name__=='__main__':
    args = parse_args()
    np.random.seed(0)
    plt.ioff()
    
    dl_epoch=300 # deep learning epoch used for HE segmentation

    patchsize = 512
    top_m = 10
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
        
    
    folder = opj(workdir, 'results', cohort, '4_non_linear_results_%s_downsampled' % args.downsample)

    # =============================================================================
    #     Start processing
    # =============================================================================
    
    for j in tqdm(tissue_df.index):
        current_data = tissue_df.iloc[j,:]
        pid, tid = tissue_df.loc[j,['pid','tid']]
        
        print('>>> pid: %s  tid: %s' % (pid, tid))
        
        resultdir = opj(folder, '%s_%d' % (pid, tid))
        if not os.path.exists(resultdir): os.makedirs(resultdir)
        
        if os.path.exists(os.path.join(resultdir,'IHC_nr_patchs','bg_ratio_summary_ptsz=%d.csv'%patchsize)):
            continue
        
        os.makedirs(opj(resultdir,'IHC_nr_patchs'), exist_ok=True)
        
        HE_tissue = Image.open(os.path.join(resultdir,'Full_resolution_HE.png'))
        HE_tissue_size = HE_tissue.size
        del HE_tissue
        flag = True
        if os.path.exists(os.path.join(resultdir,'Full_resolution_HE_seg_grey_dlepoch=%d.png' % dl_epoch)):
            HE_seg = Image.open(os.path.join(resultdir,'Full_resolution_HE_seg_grey_dlepoch=%d.png' % dl_epoch))
            if HE_tissue_size != HE_seg.size:
                x_max = max(HE_tissue_size[0], HE_seg.size[0])
                y_max = max(HE_tissue_size[1], HE_seg.size[1])
                # pad array, make sure H&E and IHC with same shape
                HE_seg_array = np.zeros((HE_tissue_size[1],HE_tissue_size[0]))
                HE_seg_array[0:HE_seg.size[1],0:HE_seg.size[0]] = np.array(HE_seg)
                HE_seg_array = HE_seg_array.astype(np.uint8)
                HE_seg = Image.fromarray(HE_seg_array)
                print('Full_resolution_HE_seg_grey has different size. Pad it.')
    #                flag = False
        else: flag = False
                
        if os.path.exists(os.path.join(resultdir,'Full_resolution_IHC_nr.png')):
            IHC_tissue = Image.open(os.path.join(resultdir,'Full_resolution_IHC_nr.png'))
            if HE_tissue_size != IHC_tissue.size:
                flag = False
                print('Invalid Full_resolution_IHC_nr')
        else: flag = False
        
        if not flag:
            print('Skip this tissue due to absence of the data.')
            continue
        
        
