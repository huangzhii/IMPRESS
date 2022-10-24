#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:08:07 2020

@author: zhihuang
"""


import sys, os, platform
print("Currently working on " + list(platform.uname())[1] + " Machine")
from tqdm import tqdm
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import glob
import pandas as pd
import random
from sklearn.model_selection import KFold
from torchvision import transforms
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
import torchvision
import copy
import argparse
from sklearn.preprocessing import MinMaxScaler
import logging
import time
from collections import Counter

def train_init(seed=0):
    torch.cuda.empty_cache()
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def nfold_cv_dict(args, workdir, n_folds=5, seed=666, testing_split=0.2):
# =============================================================================
#     n fold cross validation dictionary construction
# =============================================================================
    img_dir = workdir + 'Dataset/2_TNBC_40X_0.25MPP_%s_tiles/images/' % args.resolution
    seg_dir = workdir + 'Dataset/2_TNBC_40X_0.25MPP_%s_tiles/masks/' % args.resolution
    
    img_paths = [img_dir + p for p in os.listdir(img_dir)]
    seg_paths = [seg_dir + p for p in os.listdir(seg_dir)]
    
    barcodes = [b.split('/')[-1].split('_xmin')[0] for b in img_paths]
    barcodes_unique = np.unique(barcodes)
    barcodes_unique = np.sort(barcodes_unique)
    
    random.seed(seed)
    testing_index = random.sample(range(len(barcodes_unique)), int(len(barcodes_unique)*testing_split))
    train_val_index = [i for i in range(len(barcodes_unique)) if i not in testing_index]
    
    dataset_split = {}
    dataset_split['hold_out_testing_barcode'] = barcodes_unique[testing_index]
    dataset_split['training_validation_barcode'] = barcodes_unique[train_val_index]
    dataset_split['training_validation_5fold'] = {}
    
    kf = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
    
    for i, (train_index, val_index) in enumerate(kf.split(dataset_split['training_validation_barcode'])):
        dataset_split['training_validation_5fold']['fold ' + str(i+1)] = {}
        dataset_split['training_validation_5fold']['fold ' + str(i+1)]['train_barcode'] = dataset_split['training_validation_barcode'][train_index]
        dataset_split['training_validation_5fold']['fold ' + str(i+1)]['val_barcode'] = dataset_split['training_validation_barcode'][val_index]
    
    return dataset_split, kf


def seg_mask(color, seg):
    for c in color['exclude'] + color['other']:
        seg = np.where(seg == c, 100, seg)
    for c in color['stroma']:
        seg = np.where(seg == c, 200, seg)
    for c in color['tumor']:
        seg = np.where(seg == c, 300, seg)
    for c in color['lymphocytes']:
        seg = np.where(seg == c, 400, seg)
    seg = (seg/100).astype(int)-1
    return seg

def get_data(args,
             workdir,
             n_class,
             dataset_split,
             seg_label,
             test_type = 'test',
             fold = 1):
    
    img_dir = workdir + 'Dataset/2_TNBC_40X_0.25MPP_%s_tiles/images/' % args.resolution
    seg_dir = workdir + 'Dataset/2_TNBC_40X_0.25MPP_%s_tiles/masks/' % args.resolution
# =============================================================================
#  Get data
# =============================================================================
    img_seg_paths = pd.DataFrame(columns = ['img_path', 'seg_path', 'barcode', 'w', 'h'])
    for p in os.listdir(img_dir):
        row = {'img_path': os.path.join(img_dir, p), 'barcode': p.split('_xmin')[0],
               'w': int(p.split('_')[3]), 'h': int(p.split('_')[4])}
        img_seg_paths = img_seg_paths.append(row, ignore_index = True)
    for p in os.listdir(seg_dir):
        barcode = p.split('_xmin')[0]
        w, h = int(p.split('_')[3]), int(p.split('_')[4])
        img_seg_paths.loc[(img_seg_paths['barcode'] == barcode) & \
                          (img_seg_paths['w'] == w) & \
                          (img_seg_paths['h'] == h),'seg_path'] = os.path.join(seg_dir, p)
            
    color = {}
    color['tumor'] = [1, 19, 20]
    color['stroma'] = [2]
    color['lymphocytes'] = [3, 10, 11, 14]
    color['exclude'] = [7] # Exclude (artifacts, tears, empty lumina, etc)
    color['other'] = [c for c in seg_label.GT_code.values if c not in \
                   color['tumor'] + color['stroma'] + color['exclude'] + color['lymphocytes']]
# =============================================================================
# Train
# =============================================================================
            
    img_train, seg_train = [], []
    for i in tqdm(range(len(img_seg_paths))):
        series = img_seg_paths.iloc[i]
        img_path = series['img_path']
        seg_path = series['seg_path']
        barcode = series['barcode']
        if barcode in dataset_split['training_validation_5fold']['fold ' + str(fold)]['train_barcode']:
            img_train.append(np.array(Image.open(img_path)))
            seg_train.append(np.array(Image.open(seg_path)).astype(int))
        if test_type == 'test':
            if barcode in dataset_split['training_validation_5fold']['fold ' + str(fold)]['val_barcode']:
                img_train.append(np.array(Image.open(img_path)))
                seg_train.append(np.array(Image.open(seg_path)).astype(int))
    img_train, seg_train = np.array(img_train), np.array(seg_train)
    
    seg_train = seg_mask(color, seg_train)

    n_pixels = np.unique(seg_train, return_counts=True)[1]

    #scale the input image and swapaxes
    img_train = torch.FloatTensor(img_train)/255.0
    img_train = img_train.permute(0,3,1,2)
    seg_train = torch.ByteTensor(seg_train) # 8-bit integer (unsigned)
    train_dataset = torch.utils.data.TensorDataset(img_train, seg_train)
    del img_train, seg_train
    
# =============================================================================
#     Test
# =============================================================================
    
    img_test, seg_test = [], []
    for i in tqdm(range(len(img_seg_paths))):
        series = img_seg_paths.iloc[i]
        img_path = series['img_path']
        seg_path = series['seg_path']
        barcode = series['barcode']
        if test_type == 'test':
            if barcode in dataset_split['hold_out_testing_barcode']:
                img_test.append(np.array(Image.open(img_path)))
                seg_test.append(np.array(Image.open(seg_path)).astype(int))
        elif barcode in dataset_split['training_validation_5fold']['fold ' + str(fold)]['val_barcode']:
            img_test.append(np.array(Image.open(img_path)))
            seg_test.append(np.array(Image.open(seg_path)).astype(int))
    img_test, seg_test = np.array(img_test), np.array(seg_test)

    seg_test = seg_mask(color, seg_test)
    
    if test_type != 'test':
        n_pixels += np.unique(seg_test, return_counts=True)[1]

    #scale the input image and swapaxes
    img_test = torch.FloatTensor(img_test)/255.0
    img_test = img_test.permute(0,3,1,2)
    seg_test = torch.ByteTensor(seg_test) # 8-bit integer (unsigned)
    test_dataset = torch.utils.data.TensorDataset(img_test, seg_test)
    del img_test, seg_test
    
    weight = 1/n_pixels
    weight = weight / np.linalg.norm(weight)
    print('MSE loss weight: ', weight)

    return train_dataset, test_dataset, weight


def init_model(model_name, n_class, warmstartfrom, results_dir_dataset_fold = None):
    if model_name == 'deeplabv3':
#        model = torch.hub.load('pytorch/vision', 'fcn_resnet101', pretrained=True)
#        model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
#        for param in model.parameters(): param.requires_grad = False
        # Replace the last fully-connected layer
        # Parameters of newly constructed modules have requires_grad=True by default
        model.classifier[4] = nn.Conv2d(256, n_class, kernel_size=(1, 1), stride=(1, 1))
        model.aux_classifier[4] = nn.Conv2d(256, n_class, kernel_size=(1, 1), stride=(1, 1))
    
    if model_name == 'unet':
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', \
                               in_channels=3, out_channels=1, init_features=32, pretrained=True)
        model.conv = nn.Conv2d(32, n_class, kernel_size=(1, 1), stride=(1, 1))
        
        
    if warmstartfrom > 0:
        mdlpath = results_dir_dataset_fold+'/mode_state_dict_epoch_' + str(warmstartfrom) + '.mdl'
        if not os.path.isfile(mdlpath):
            print(mdlpath)
            print('warm start does not have the associated model!')
            raise Exception
        else:
            model.load_state_dict(torch.load(mdlpath))
        
    if torch.cuda.is_available():
        model = model.cuda()
    return model


# create logger
def create_logger(logfile, TIMESTRING, warmstartfrom=0):
    if warmstartfrom > 0 and os.path.isfile(logfile):
        logger = logging.getLogger(logfile)
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(logfile, mode='a')
    else:
        logger = logging.getLogger(TIMESTRING)
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(logfile, mode='w')
        
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger
