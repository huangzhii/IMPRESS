#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 20:27:04 2022

@author: zhihuang
"""



import os,sys,platform
import numpy as np
import pandas as pd
import openslide
from PIL import Image
import matplotlib.pyplot as plt
from imshowpair import imshowpair
import pickle
import argparse
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

from python_script import get_data
from python_script import plots

'''
pip install numpy==1.18.1
pip install SimpleITK
pip install imshowpair
pip install scikit-image
pip install opencv-python
'''


opj = os.path.join

workdir = 'workdir'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=str, default='HER2+')
    parser.add_argument('--downsample', type=str, default='32x')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--resolution', default='20x', type=str)
    parser.add_argument('--dl_epoch', type=int)
    parser.add_argument('--pid_tid', type=str, default='R7_5')
    return parser.parse_args()
    
if __name__=='__main__':
    args = parse_args()
    np.random.seed(0)
    plt.ioff()
    
    dl_epoch = args.dl_epoch
    cohort = args.cohort
    pid_tid = args.pid_tid
    

    dlmodeldir = opj(workdir, 'HE_segmentation/deeplabv3/20x_40X_0.25MPP/model.mdl')


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
        
        # if ('%s_%s' % (pid,tid)) != args.pid_tid:
        #     continue
        print('>>> pid: %s  tid: %s' % (pid, tid))
        
        resultdir = opj(folder, '%s_%d' % (pid, tid))
        if not os.path.exists(resultdir): os.makedirs(resultdir)

        # =============================================================================
        #     Get current data
        # =============================================================================
        
        HE_tissue = Image.open(os.path.join(resultdir,'Full_resolution_HE.png'))
        HE_tissue = np.array(HE_tissue)
        if os.path.exists(os.path.join(resultdir,'Full_resolution_HE_seg_grey_dlepoch=%d.png' % dl_epoch)):
            seg = Image.open(os.path.join(resultdir,'Full_resolution_HE_seg_grey_dlepoch=%d.png' % dl_epoch))
            if HE_tissue[:,:,0].T.shape == seg.size:
                print('Result existed. Exit.')
                # exit()
                pass
        fullres_size = (HE_tissue.shape[1], HE_tissue.shape[0])
        downsampling_ratio = 4
        new_size = (int(HE_tissue.shape[1]/downsampling_ratio), int(HE_tissue.shape[0]/downsampling_ratio))
        # the new_size is used to save 4x low resolution image for visualization, not for deep learning.
        im = Image.fromarray(HE_tissue)
        del HE_tissue


        # =============================================================================
        #     DL segmentation
        # =============================================================================
            
            
        n_class = 4
        patchsize = 1024
        overlap = 200
        if args.resolution == '20x':
            magnification = 40
            downsample = magnification / int(args.resolution.split('x')[0])
        elif args.resolution == '40x':
            downsample = 1

        img_test_all = []
        im = im.resize((int(im.size[0]/downsample), int(im.size[1]/downsample)))
        imgwidth, imgheight = im.size
        for i in range(0,imgheight,patchsize-overlap):
            for j in range(0,imgwidth,patchsize-overlap):
                box = (j, i, j+patchsize, i+patchsize)
                patch = im.crop(box)
                patch = np.array(patch)/255.0
                img_test_all.append(patch.astype(np.float16))
        img_test_all = np.stack(img_test_all, axis=0)
        img_test_all = img_test_all.swapaxes(2,3).swapaxes(1,2)
        img_test_all = img_test_all.astype(np.float16)
        seg_pred_map_per_class = np.zeros((im.size[1], im.size[0], n_class)).astype(np.float16)
        
        
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        # Replace the last fully-connected layer
        # Parameters of newly constructed modules have requires_grad=True by default
        model.classifier[4] = nn.Conv2d(256, n_class, kernel_size=(1, 1), stride=(1, 1))
        model.aux_classifier[4] = nn.Conv2d(256, n_class, kernel_size=(1, 1), stride=(1, 1))
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(dlmodeldir))
            model.cuda()
        else:
            model.load_state_dict(torch.load(dlmodeldir, map_location=torch.device('cpu')))
        model.eval()
        
        
        idx_df = pd.DataFrame()
        for i in range(0,imgheight,patchsize-overlap):
            for j in range(0,imgwidth,patchsize-overlap):
                x1 = i
                x2 = min(i+patchsize, imgheight)
                y1 = j
                y2 = min(j+patchsize, imgwidth)
                row = pd.DataFrame({'i':i, 'j':j}, index=[0])
                idx_df = pd.concat([idx_df, row], ignore_index=True)
                
        idx=-1
        for img in tqdm(img_test_all):
            idx+=1
            img = torch.FloatTensor(img).unsqueeze(0)
            if torch.cuda.is_available():
                img = img.cuda()
            output = model(img)
            seg_pred, seg_aux = output['out'], output['aux']
            seg_pred = seg_pred.data.cpu().numpy().astype(np.float16)
            if seg_pred.shape[0] != 1:
                print('seg_pred.shape[0] != 1')
                exit()
            seg_pred_proba = seg_pred.swapaxes(1,2).swapaxes(2,3)[0,:,:,:]
            i, j = idx_df.iloc[idx,:].astype(int)
            x1 = i
            x2 = min(i+patchsize, imgheight)
            y1 = j
            y2 = min(j+patchsize, imgwidth)
            seg_pred_map_per_class[x1:x2, y1:y2, :] += seg_pred_proba[0:x2-x1, 0:y2-y1, :]
        
        
        seg_pred_map = np.argmax(seg_pred_map_per_class, axis = 2)
        
        
        del model, img_test_all

        print('Feedfoward done.')


            
        # =============================================================================
        #     Save figures
        # =============================================================================
            
        seg_pred_map_fullres = Image.fromarray(np.uint8(seg_pred_map)).resize(fullres_size)
        seg_pred_map_fullres.save(opj(resultdir, 'Full_resolution_HE_seg_grey_dlepoch=%d.png' % dl_epoch))
        del seg_pred_map_fullres
        
        seg_pred_map_color = np.zeros((seg_pred_map.shape[0], seg_pred_map.shape[1], 3)).astype(np.uint8)
        seg_pred_map_color[seg_pred_map == 0, :] = [255, 255, 255] # exclude: white
        seg_pred_map_color[seg_pred_map == 1, :] = [242, 211, 232] # stroma: pink
        seg_pred_map_color[seg_pred_map == 2, :] = [125, 62, 173] # tumor: violet
        seg_pred_map_color[seg_pred_map == 3, :] = [96, 113, 209] # lymphocytes: blue
        
        
        seg_pred_map_color_fullres = Image.fromarray(np.uint8(seg_pred_map_color)).resize(fullres_size)
        del seg_pred_map_color
        
        seg_pred_map_color_fullres.save(opj(resultdir, 'Full_resolution_HE_seg_color_dlepoch=%d.png' % dl_epoch))
        
        
        # save low resolution and fused image
        seg_pred_map_color_fullres.resize(new_size).save(opj(resultdir, 'Downsample=%dx_HE_seg_dlepoch=%d.png' % \
                                                             (downsampling_ratio, dl_epoch)))
        
            
        print('Successfully generate the results.')
        
