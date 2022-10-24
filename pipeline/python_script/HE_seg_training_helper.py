#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:25:51 2019

@author: zhihuang
"""

import sys, os, platform
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
from torch.autograd import Variable
import torchvision
import copy
import argparse
from sklearn.preprocessing import MinMaxScaler
import logging
import time
import torch.utils.data
from apex import amp


def criterion(input, target, weight):
    weight = weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(target)
    return torch.sum(weight * (input - target) ** 2)
# criterion = nn.MSELoss()
# if torch.cuda.is_available():
#     criterion = criterion.cuda()


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def compute_hist(label_trues, label_preds, n_class):
    label_trues = np.argmax(label_trues, axis=1)
    label_preds = np.argmax(label_preds, axis=1)
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    return hist


def label_accuracy_score(hist):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    with np.errstate(divide='ignore', invalid='ignore'):
        dice = 2*np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0))
    # Jaccard index is also known as Intersection over Union (IoU)
    mean_iu = np.nanmean(iu)
    mean_dice = np.nanmean(dice)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, iu, mean_iu, dice, mean_dice, fwavacc
    
def training_validating_models(model, optimizer, args,  class_weight,
                               train_dataset, test_dataset,
                               n_class, logger, \
                               results_dir_dataset, model_name, auxiliary_loss_weight = 0.5, save_image_per_epoch = 10, \
                               save_model_per_epoch = 50, verbose = 0):
    print('Cuda is available? ---- %s' % str(torch.cuda.is_available()))
    '''
    Native PyTorch automatic mixed precision for faster training on NVIDIA GPUs
    https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
    '''
    if args.amp:
        # Initialization
        opt_level = 'O1'
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    #    scaler = torch.cuda.amp.GradScaler() # automatic mixed precision
    class_weight = torch.FloatTensor(class_weight)
    if torch.cuda.is_available():
        class_weight = class_weight.cuda()
    # Parameters
    dataloader_params = {'batch_size': args.batch_size,
                          'shuffle': True,
                          'num_workers': args.workers,
                          'pin_memory': False,
                          'drop_last': False} #Use drop_last to prevent during training the current batch only contains a single sample
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_params)
    
    dataloader_params = {'batch_size': args.batch_size,
                          'shuffle': False,
                          'num_workers': args.workers,
                          'pin_memory': False,
                          'drop_last': False} #Use drop_last to prevent during training the current batch only contains a single sample
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_params)
    
    
    training_result = pd.DataFrame(columns = ['epoch','loss','acc','acc_cls','fwavacc','mean_iu','iu','mean_dice','dice'])
    validation_result = pd.DataFrame(columns = ['epoch','loss','acc','acc_cls','fwavacc','mean_iu','iu','mean_dice','dice'])
    best_validation_mean_iu = 0
    """Returns accuracy score evaluation result.
      - loss
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    for idx in tqdm(range(args.warmstartfrom, args.nepochs)):
        epoch = idx + 1
        loss_per_epoch = 0
        model.train()
        img, seg = None, None
        hist = np.zeros((n_class, n_class))
        for img, seg in train_loader:
            seg = torch.nn.functional.one_hot(seg.long(), num_classes=n_class).permute(0,3,1,2)
            if torch.cuda.is_available():
                img, seg = img.cuda(), seg.cuda()
            optimizer.zero_grad()
            
            if False:
                with torch.cuda.amp.autocast(): 
                    output = model(img)
                    del img
                    if model_name =='deeplabv3':
                        seg_pred, seg_aux = output['out'], output['aux']
            #                seg_pred = seg_pred.argmax(1).float()
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        loss1 = criterion(seg_pred, seg, class_weight)
                        loss2 = criterion(seg_aux, seg, class_weight)
                        loss = loss1 + auxiliary_loss_weight*loss2
            else:
                output = model(img)
                del img
                if model_name =='deeplabv3':
                    seg_pred, seg_aux = output['out'], output['aux']
        #                seg_pred = seg_pred.argmax(1).float()
                    # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                    loss1 = criterion(seg_pred, seg, class_weight)
                    loss2 = criterion(seg_aux, seg, class_weight)
                    loss = loss1 + auxiliary_loss_weight*loss2
            if args.amp:
                '''
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                '''
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
            else:
                loss.backward()
                optimizer.step()
                
            loss_per_epoch += loss.data.cpu().numpy()
            seg_pred_int = np.argmax(seg_pred.data.cpu().numpy(), axis = 1)
            seg_pred_int = (np.arange(n_class) == seg_pred_int[...,None]).astype(int)
            seg_pred_int = seg_pred_int.swapaxes(2,3).swapaxes(1,2) # one hot encode
            seg_int = seg.data.cpu().numpy().astype(int)
            hist += compute_hist(seg_int, seg_pred_int, n_class)
            del seg
            
        torch.cuda.empty_cache()
        
        acc, acc_cls, iu, mean_iu, dice, mean_dice, fwavacc = label_accuracy_score(hist)
        training_result = training_result.append({'epoch': epoch,'loss': loss_per_epoch,\
                                                  'acc': acc,'acc_cls': acc_cls, 'fwavacc': fwavacc,\
                                                  'mean_iu': mean_iu, 'iu': str(iu),\
                                                  'mean_dice': mean_dice, 'dice': str(dice)
                                                  }, ignore_index=True)
                
        logger.log(logging.INFO, "[%d/%d] [Training]: loss: %f, acc: %f, acc_cls: %f, fwavacc: %f, mean_iu: %f, iu: %s, mean_dice: %f, dice: %s" % \
                   (epoch, args.nepochs, loss_per_epoch, acc, acc_cls, fwavacc, mean_iu, str(iu), mean_dice, str(dice)))
        if verbose > 0:
            print("[%d/%d] [Training]: loss: %f, acc: %f, acc_cls: %f, fwavacc: %f, mean_iu: %f, iu: %s, mean_dice: %f, dice: %s" % \
                       (epoch, args.nepochs, loss_per_epoch, acc, acc_cls, fwavacc, mean_iu, str(iu), mean_dice, str(dice)))
        with amp.disable_casts():
            with torch.no_grad():
                loss_per_epoch = 0
                model.eval()
                img, seg = None, None
                hist = np.zeros((n_class, n_class))
                for img, seg in test_loader:
                    seg = torch.nn.functional.one_hot(seg.long(), num_classes=n_class).permute(0,3,1,2)
                    if torch.cuda.is_available():
                        img, seg = img.cuda(), seg.cuda()
                    output = model(img)                
                    if model_name =='deeplabv3':
                        seg_pred, seg_aux = output['out'], output['aux']
            #                seg_pred = seg_pred.argmax(1).float()
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        loss1 = criterion(seg_pred, seg, class_weight)
                        loss2 = criterion(seg_aux, seg, class_weight)
                        loss = loss1 + auxiliary_loss_weight*loss2
                        
                    loss_per_epoch += loss.data.cpu().numpy()
                    seg_pred_int = np.argmax(seg_pred.data.cpu().numpy(), axis = 1)
                    seg_pred_int = (np.arange(n_class) == seg_pred_int[...,None]).astype(int).swapaxes(2,3).swapaxes(1,2) # one hot encode
                    seg_int = seg.data.cpu().numpy().astype(int)
                    hist += compute_hist(seg_int, seg_pred_int, n_class)
                    
            acc, acc_cls, iu, mean_iu, dice, mean_dice, fwavacc = label_accuracy_score(hist)
            validation_result = validation_result.append({'epoch': epoch,'loss': loss_per_epoch,\
                                                      'acc': acc,'acc_cls': acc_cls, 'fwavacc': fwavacc,\
                                                      'mean_iu': mean_iu, 'iu': str(iu),\
                                                      'mean_dice': mean_dice, 'dice': str(dice)
                                                      }, ignore_index=True)
            logger.log(logging.INFO, "[%d/%d] Validation: loss: %f, acc: %f, acc_cls: %f, fwavacc: %f, mean_iu: %f, iu: %s, mean_dice: %f, dice: %s" % \
                       (epoch, args.nepochs, loss_per_epoch, acc, acc_cls, fwavacc, mean_iu, str(iu), mean_dice, str(dice)))
            if verbose > 0:
                print("[%d/%d] Validation: loss: %f, acc: %f, acc_cls: %f, fwavacc: %f, mean_iu: %f, iu: %s, mean_dice: %f, dice: %s" % \
                           (epoch, args.nepochs, loss_per_epoch, acc, acc_cls, fwavacc, mean_iu, str(iu), mean_dice, str(dice)))
    
        # deep copy the model
        if mean_iu > best_validation_mean_iu:
            best_validation_mean_iu = mean_iu
            best_model = copy.deepcopy(model.state_dict())
            
        if epoch % save_image_per_epoch == 0:
            last_img = (np.array(img[-1,:,:,:].data.cpu().numpy()).swapaxes(0,1).swapaxes(1,2)*255).astype(np.uint8)
            last_seg = np.array(seg.argmax(1)[-1,:,:].data.cpu().numpy()*np.floor(255/(n_class-1))).astype(np.uint8)
            last_seg = np.dstack([last_seg]*3)
            last_seg_pred = np.array(seg_pred.argmax(1)[-1,:,:].data.cpu().numpy()*np.floor(255/(n_class-1))).astype(np.uint8)
            last_seg_pred = np.dstack([last_seg_pred]*3)
            
            concat_img = np.concatenate( (last_img, last_seg, last_seg_pred), axis=1)
            Image.fromarray(concat_img).save(results_dir_dataset + '/epoch_' + str(epoch) + '_img_seg_pred.png')
            
        if epoch % save_model_per_epoch == 0:
            torch.save(model.state_dict(), results_dir_dataset + '/mode_state_dict_epoch_' + str(epoch) + '.mdl')
            pd.DataFrame(hist).to_csv(results_dir_dataset + '/epoch_' + str(epoch) + '_pred_hist' + '.csv')

        del img, seg
        torch.cuda.empty_cache()
        last_model = model
            
    return training_result, validation_result, best_model, last_model




def plot_figure(training_result, validation_result, fname, lr):
    plt.figure(figsize=(15,10))
    plt.suptitle('Plot with LR = ' + str(lr), fontsize=14, fontweight='bold')
    
    plt.subplot(2, 2, 1)
    plt.plot(training_result.epoch.values, training_result.loss.values, '-')
    plt.subplots_adjust(top=0.8)
    plt.title("Training Loss")
    plt.ylabel('Loss')
    
    plt.subplot(2, 2, 2)
    plt.plot(validation_result.epoch.values, validation_result.loss.values, '-')
    plt.subplots_adjust(top=0.8)
    plt.title("Validation Loss")
    plt.ylabel('Loss')
    
    plt.subplot(2, 2, 3)
    plt.plot(training_result.epoch.values, training_result.mean_iu.values, '-')
    plt.subplots_adjust(top=0.8)
    plt.title("Training Mean IoU")
    plt.ylabel('Mean IoU')
    
    plt.subplot(2, 2, 4)
    plt.plot(validation_result.epoch.values, validation_result.mean_iu.values, '-')
    plt.subplots_adjust(top=0.8)
    plt.title("Validation Mean IoU")
    plt.ylabel('Mean IoU')
    plt.savefig(fname, dpi = 300)

