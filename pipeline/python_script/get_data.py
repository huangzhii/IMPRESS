#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:06:40 2020

@author: zhihuan
"""
import os
import numpy as np
import pandas as pd
import openslide
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 9933120000

from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt

opj = os.path.join



def get_patient_list(datadir, cohort='HER2+'):
    # cohort: HER2+, TNBC
    patient_list = pd.DataFrame(columns = ['pid', 'path2HE', 'path2IHC'])
    
    if cohort == 'HER2+':
        flist = [v for v in os.listdir(datadir[cohort]) if ('HER2' in v) and ('H&E' in v)]
    elif cohort == 'TNBC':
        flist = [v for v in os.listdir(datadir[cohort]) if ('TNBC' in v) and ('H&E' in v)]
    else:
        raise Exception('%s cohort is not available. Exit.' % cohort)
        
    for f in flist:
        pid = f.split('-')[1].split(' ')[0]
        path2HE = opj(datadir[cohort], f)
        path2IHC = opj(datadir[cohort], f.replace('H&E','IHC'))
        row = pd.DataFrame([[pid, path2HE, path2IHC]], columns = ['pid', 'path2HE', 'path2IHC'])
        patient_list = pd.concat([patient_list, row], axis=0, ignore_index=True)
    
    patient_list = patient_list.sort_values('pid', ignore_index=True)
    return patient_list
        
def get_tissue_list(tissuedir, cohort='HER2+'):
    # cohort: HER2+, TNBC
    tissue_list = pd.DataFrame(columns = ['pid', 'tid', 'path2HE', 'path2IHC'])
    
    for root, dirs, files in os.walk(tissuedir[cohort]):
        for file in files:
            if file.endswith("_H&E.png"):
                pid = file.split('_')[0]
                tid = int(file.split('_')[1])
                row = pd.DataFrame({'pid': pid,'tid': tid,
                                    'path2HE': os.path.join(root, file)}, index=[0])
                tissue_list = pd.concat([tissue_list, row], ignore_index = True)
             
    for root, dirs, files in os.walk(tissuedir[cohort]):
        for file in files:
            if file.endswith("_IHC.png"):
                pid = file.split('_')[0]
                tid = int(file.split('_')[1])
                tissue_list.loc[(tissue_list['pid'] == pid) & \
                                (tissue_list['tid'] == tid), 'path2IHC'] = os.path.join(root, file)
    tissue_list = tissue_list.sort_values(['pid', 'tid'], ignore_index=True)
    return tissue_list
    
def SVS_tissue_extraction(wsi, polygon, bg_color = 243):
    #polygon should be in the original 40x coordinates.
    x1, x2 = int(np.min(polygon[:,0])), int(np.max(polygon[:,0]))
    y1, y2 = int(np.min(polygon[:,1])), int(np.max(polygon[:,1]))
    
    location = (x1, y1)
    size = (x2-x1, y2-y1)
    tissue = wsi.read_region(location=location, level=0, size=size)
    
    new_x1, new_x2 = x1-x1, x2-x1
    new_y1, new_y2 = y1-y1, y2-y1
    
    new_polygon = [(a-x1, b-y1) for a,b in polygon.astype(np.int32)]
    imArray = np.asarray(tissue)
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(new_polygon, outline=1, fill=1)
    mask = np.array(maskIm)
    # assemble new image (uint8: 0-255)
    newImArray = np.empty(imArray.shape,dtype='uint8')
    # colors (three first columns, RGB)
    newImArray[:,:,:3] = imArray[:,:,:3]
    # transparency (4th column)
    newImArray[:,:,3] = mask*255
    # bounding box
    newImArray = newImArray[new_y1:new_y2, new_x1:new_x2, :] # need to inverse x and y
    newImArray[newImArray[...,3]==0, :] = (bg_color,bg_color,bg_color,255)
    # back to Image from numpy
    # newIm = Image.fromarray(newImArray, "RGBA")
    # newIm.resize((600,200)).show()
    # maskIm.resize((600,200)).show()
    return newImArray[..., :3]



def get_tissue_image(args, cohort, pid, tid, patient_df, tissue_df, get_ihc = True):
    
    series = tissue_df.loc[(tissue_df['pid'] == pid) & (tissue_df['tid'] == tid),:]
    path2HE_tissue = series['path2HE'].values[0]
    path2IHC_tissue = series['path2IHC'].values[0]
    
    series = patient_df.loc[patient_df['pid'] == pid,:]
    path2HE_WSI = series['path2HE'].values[0]
    path2IHC_WSI = series['path2IHC'].values[0]

    properties = {}
    HE_WSI = openslide.OpenSlide(path2HE_WSI)
    properties['H&E'] = dict(HE_WSI.properties)
    print('H&E Magnification: %s   MPP: %s' % (properties['H&E']['aperio.AppMag'], properties['H&E']['aperio.MPP']))
    HE_size = HE_WSI.level_dimensions[0]
    
    IHC_WSI = openslide.OpenSlide(path2IHC_WSI)
    properties['IHC'] = dict(IHC_WSI.properties)
    print('IHC Magnification: %s   MPP: %s' % (properties['IHC']['aperio.AppMag'], properties['IHC']['aperio.MPP']))
    IHC_size = IHC_WSI.level_dimensions[0]
    
    
    polygon_dir = os.path.dirname(path2HE_tissue.replace('3_tissues_%s_downsampled/tissue' % args.downsample, '2_ImageLabeler_Polygon_%s_downsampled' % args.downsample))
    
    
    curr_polygon_path_HE = opj(polygon_dir, '%s_H&E_polygon.csv' % pid)
    polygon_HE = pd.read_csv(opj(curr_polygon_path_HE), header=None, index_col=0)
    polygon_HE.columns = ['x', 'y']
    polygon_HE = polygon_HE.loc[tid,:].values * int(args.downsample.rstrip('x'))
    HE_tissue = SVS_tissue_extraction(HE_WSI, polygon_HE, bg_color=args.bg_color)
    
    curr_polygon_path_IHC = opj(polygon_dir, '%s_IHC_polygon.csv' % pid)
    polygon_IHC = pd.read_csv(opj(curr_polygon_path_IHC), header=None, index_col=0)
    polygon_IHC.columns = ['x', 'y']
    polygon_IHC = polygon_IHC.loc[tid,:].values * int(args.downsample.rstrip('x'))
    IHC_tissue = SVS_tissue_extraction(IHC_WSI, polygon_IHC, bg_color=args.bg_color)


    x_max = max(HE_tissue.shape[0], IHC_tissue.shape[0])
    y_max = max(HE_tissue.shape[1], IHC_tissue.shape[1])
    
    # pad array, make sure H&E and IHC with same shape
    HE_tissue = np.pad(HE_tissue, pad_width = [(0, x_max-HE_tissue.shape[0]), (0, y_max-HE_tissue.shape[1]), (0, 0)], constant_values = args.bg_color)
    IHC_tissue = np.pad(IHC_tissue, pad_width = [(0, x_max-IHC_tissue.shape[0]), (0, y_max-IHC_tissue.shape[1]), (0, 0)], constant_values = args.bg_color)
    
    return HE_tissue, IHC_tissue, properties

def get_landmarks(tid, tissue_df, landmarkdir):
    series = tissue_df.iloc[tid,:]
    file = '%s%03d_%02d.csv' % (landmarkdir, series['pid'], series['tid'])
    if os.path.exists(file):
        landmarks = pd.read_csv(file)
    else:
        print('No landmark found. Use None type instead.')
        landmarks = None
    return landmarks
    