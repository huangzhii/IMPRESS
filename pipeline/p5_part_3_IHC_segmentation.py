#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 21:30:32 2022

@author: zhihuang
"""


# Part 3: apply



import os,sys,platform
import numpy as np
import pandas as pd
from PIL import Image, ImageCms
Image.MAX_IMAGE_PIXELS = 9933120000
import matplotlib.pyplot as plt
from imshowpair import imshowpair
import pickle
import argparse
from tqdm import tqdm

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


distinct_colors = [
        "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
        "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
        "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
        "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
        "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",

        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
        "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
        "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
        "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
        "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
        "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
        "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
        "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"
]
# HEX to RGB
distinct_colors = [tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for h in distinct_colors]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=str, default='TNBC', choices=['HER2+', 'TNBC'])
    parser.add_argument('--downsample', type=str, default='32x')
    parser.add_argument('--memory_efficient', type=bool, default=True)
    parser.add_argument('--pid_tid', type=str, default='R3_2')
    parser.add_argument('--k', type=int)
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    np.random.seed(0)
    plt.ioff()
    cohort = args.cohort
    patchsize = 512
    color.colorconv.lab_ref_white = np.array([0.96422, 1.0, 0.82521])
    k = args.k
    
    colors = distinct_colors[1:k+1]
    

    datadir = opj(workdir, '..', 'Zhi-Validation set/')
    tissuedir = opj(workdir, 'results', args.cohort, '3_tissues_32x_downsampled', 'tissue')
    kmeans_save_dir = opj(workdir, 'results', args.cohort, 'IHC_kmeans')

    # =============================================================================
    #         # Determine the object centers
    # =============================================================================
    with open(os.path.join(kmeans_save_dir, 'K=%d_kmeansModel.pkl' % k), 'rb') as f:
        kmeans = pickle.load(f)
    with open(os.path.join(kmeans_save_dir, 'object_centers_K=%d.txt' % k),'r') as f:
        line = f.readlines()
        CD8_centers = re.findall('\[.*\]',line[1])[0][1:-1].split(', ')
        CD163_centers = re.findall('\[.*\]',line[2])[0][1:-1].split(', ')
        PDL1_centers = re.findall('\[.*\]',line[3])[0][1:-1].split(', ')
        CD8_centers = [int(c) for c in CD8_centers]
        CD163_centers = [int(c) for c in CD163_centers]
        PDL1_centers = [int(c) for c in PDL1_centers]
        object_centers = {'CD8':CD8_centers,
                          'CD163':CD163_centers,
                          'PDL1':PDL1_centers}
        
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
    for j in tqdm(tissue_df.index):
        current_data = tissue_df.iloc[j,:]
        pid, tid = tissue_df.loc[j,['pid','tid']]
        
        if ('%s_%s' % (pid,tid)) != args.pid_tid:
            continue
        print('>>> pid: %s  tid: %s' % (pid, tid))
        
        resultdir = opj(folder, '%s_%d' % (pid, tid))
        if not os.path.exists(resultdir): os.makedirs(resultdir)
        
        if os.path.exists(os.path.join(resultdir, 'Full_resolution_IHC_nr_kmeans_seg_k=%d.png' % k)):
            continue
        
        # Get IHC_nr results
        print('Get IHC_nr results')
        IHC_tissue = Image.open(os.path.join(resultdir,'Full_resolution_IHC_nr.png'))
        # assign labels to pixels (L*a*b*)
        print('Assign labels to pixels (L*a*b*)')
        IHC_size = IHC_tissue.size
        IHC_tissue = np.array(IHC_tissue)
        IHC_tissue = IHC_tissue.astype(np.uint8)
        
        if args.memory_efficient:
            n_part = 100
            labels = None
            for i in tqdm(range(n_part)):
                x1, x2 = int(IHC_tissue.shape[0]*i/n_part), int(IHC_tissue.shape[0]*(i+1)/n_part)
                img_part = IHC_tissue[x1:x2, ...]
                pv_lab_train_part = color.rgb2lab(img_part).reshape(-1,3)
                labels_part = kmeans.predict(pv_lab_train_part).flatten()
                if labels is None:
                    labels = labels_part
                else:
                    labels = np.concatenate([labels, labels_part], axis=0)
        else:
            pv_lab_train = color.rgb2lab(IHC_tissue).reshape(-1,3)
            labels = kmeans.predict(pv_lab_train)
            labels = labels.flatten()
            del pv_lab_train
        


        labels_image = np.repeat(labels[:, np.newaxis], 3, axis=1).reshape((IHC_size[1],IHC_size[0],3))
        del labels
        del IHC_tissue
        
    # =============================================================================
    #         # show the image with segmentation mask
    # =============================================================================
            
        object_colors = {'CD8':distinct_colors[5],'CD163':distinct_colors[4],'PDL1':distinct_colors[9]}
        
        object_image = np.ones((IHC_size[1],IHC_size[0],3))*255
        for cell in object_centers:
            for v in object_centers[cell]:
                object_image[labels_image[:,:,0] == v,:] = object_colors[cell]
        object_image = Image.fromarray(object_image.astype(np.uint8))
        downsampling = 4
        downsampling_size = (round(IHC_size[0]/downsampling), round(IHC_size[1]/downsampling))
        object_image_downsampled = object_image.resize(downsampling_size)
        object_image_downsampled.save(os.path.join(resultdir, 'Downsample=%dx_IHC_nr_kmeans_seg_color_k=%d.png' % (downsampling, k)))
        del object_image_downsampled
        
        seg_pred = np.zeros((IHC_size[1],IHC_size[0]))
        for cell in object_centers:
            for v in object_centers[cell]:
                if cell == 'CD8': value=1
                if cell == 'CD163': value=2
                if cell == 'PDL1': value=3
                seg_pred[labels_image[:,:,0] == v] = value
        seg_pred = Image.fromarray(seg_pred.astype(np.uint8))
        seg_pred.save(os.path.join(resultdir, 'Full_resolution_IHC_nr_kmeans_seg_k=%d.png' % k))
        del seg_pred
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
