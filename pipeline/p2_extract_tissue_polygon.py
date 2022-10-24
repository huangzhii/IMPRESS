#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 19:59:43 2022

@author: zhihuang
"""


import os, sys, platform
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
opj = os.path.join


def img_tissue_extraction(img, polygon, bg_color=240):
    # convert to numpy (for convenience)
    imArray = np.asarray(img)
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
    mask = np.array(maskIm)
    # assemble new image (uint8: 0-255)
    newImArray = np.empty(imArray.shape,dtype='uint8')
    # colors (three first columns, RGB)
    newImArray[:,:,:3] = imArray[:,:,:3]
    # transparency (4th column)
    newImArray[:,:,3] = mask*255
    # bounding box
    x_min, x_max = np.min([v1 for v1,v2 in polygon]), np.max([v1 for v1,v2 in polygon])
    y_min, y_max = np.min([v2 for v1,v2 in polygon]), np.max([v2 for v1,v2 in polygon])
    newImArray = newImArray[y_min:y_max, x_min:x_max, :] # need to inverse x and y
    newImArray[newImArray[...,3]==0, :] = (bg_color,bg_color,bg_color,255)
    # back to Image from numpy
    newIm = Image.fromarray(newImArray, "RGBA")
    return newIm


if __name__ == '__main__':
    resultdir = 'results'

    downsample = '32x'
    
    for cohort in ['HER2+']:
        resultfolder = opj(resultdir, cohort, '3_tissues_%s_downsampled' % downsample, 'tissue')
        
        img_dir = opj(resultdir, cohort, '1_%s_downsampled' % downsample)
        polygon_dir = opj(resultdir, cohort, '2_ImageLabeler_Polygon_%s_downsampled' % downsample)
        downsample_imgs = os.listdir(img_dir)
        downsample_polygons = os.listdir(polygon_dir)
        
        for f in tqdm(downsample_imgs):
            
            id, mode = f.rstrip('.png').split('_')
            polygon = pd.read_csv(opj(polygon_dir, '%s_%s_polygon.csv' % (id, mode) ), header=None)
            polygon.columns = ['tissue_ID', 'x', 'y']
            
            img = Image.open(opj(img_dir, f)).convert("RGBA")
            
            for tissue_id in polygon['tissue_ID'].unique():
                this_polygon = polygon.loc[polygon['tissue_ID']==tissue_id, ['x','y']].values.round().astype(np.int32)
                this_polygon = [(a,b) for a,b in this_polygon]
                tissue = img_tissue_extraction(img, this_polygon)
                f_out = '%s_%d_%s.png' % (id, tissue_id, mode)
                tissue.save(opj(resultfolder, f_out))
        
        
        
        
        
        
        
        
        
        
        
        
