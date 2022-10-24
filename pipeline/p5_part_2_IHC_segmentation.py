#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 20:27:04 2022

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

import time
import cv2
import sklearn
from sklearn.cluster import KMeans, MiniBatchKMeans
from skimage import color
import random
import glob
import copy
'''
pip install numpy==1.18.1
pip install SimpleITK
pip install imshowpair
pip install scikit-image
pip install opencv-python
'''

opj = os.path.join

workdir = 'workdir'
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

         
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def compute_hist(label_trues, label_preds, n_class): # same results as sklearn confusion matrix
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    return hist

def create_kmeans_model(k, minibatch, batch_size):
    if minibatch:
        kmeans=MiniBatchKMeans(n_clusters=k,
                              n_init=3,
                              batch_size=batch_size,
                              max_iter=300,
                              tol=0.0001,
                              random_state=0,
                              verbose=1)
    else:
        kmeans=KMeans(n_clusters=k,
                        n_init=3,
                        max_iter=300,
                        tol=0.0001,
                        random_state=0,
                        verbose=1)
    return kmeans
def search_kmeans(save_dir, pv_lab_train, search_k=True, maxk=50, minibatch=True, batch_size=100):
    # =============================================================================
    #     Perform K-Means
    # =============================================================================
    # The Elbow Method is one of the most popular methods to determine this optimal value of k.
    md=[]
    if search_k:
        k_range = np.arange(2,maxk+1)
        for i in tqdm(k_range):
            kmeans = create_kmeans_model(i, minibatch, batch_size)
            kmeans.fit(pv_lab_train)
            o=kmeans.inertia_ # Sum of squared distances of samples to their closest cluster center.
            md.append(o)
        print(md)
        plt.figure()
        plt.plot(list(k_range),md)
        plt.title('Optimal value of K in K-means by inertia')
        plt.ylabel('inertia')
        plt.xlabel('K')
        plt.xticks(k_range)
        plt.savefig(os.path.join(save_dir, 'K_vs_inertia.png'),dpi=600)
    else:
        kmeans = create_kmeans_model(maxk, minibatch, batch_size)
        kmeans.fit(pv_lab_train)
        inertia=kmeans.inertia_ # Sum of squared distances of samples to their closest cluster center.
        with open(os.path.join(save_dir, 'K=%d_inertia.txt' % maxk),'w+') as f:
            f.write('Inertia: %.8f' % inertia)
        kmeans.labels_ = None
        with open(os.path.join(save_dir, 'K=%d_kmeansModel.pkl' % maxk),'wb') as f:
            pickle.dump(kmeans, f)
    return md, kmeans

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

def crop(input, height, width, tile_overlapping):
    patches=[]
    im = Image.open(input)
    w, h = im.size
    num_w = int(np.floor(w/(1-tile_overlapping)/width))+1
    num_h = int(np.floor(h/(1-tile_overlapping)/height))+1
    for i in range(num_h):
        for j in range(num_w):
            start_w, start_h = int(j*width*(1-tile_overlapping)), int(i*height*(1-tile_overlapping))
            end_w, end_h = int(start_w+width), int(start_h+height)
            if end_w > w-1: start_w, end_w = w - width, w
            if end_h > h-1: start_h, end_h = h - height, h
                
            box = (start_w, start_h, end_w, end_h)
            a = im.crop(box)
            patches.append(np.array(a))
    return np.stack(patches)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=str, default='TNBC')
    parser.add_argument('--search_k', type=int, default=0, help='if > 0, use k=search_k.', choices=[0, 30])
    parser.add_argument('--minibatchKmeans', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--dl_epoch', type=int)
    parser.add_argument('--k', type=int)
    parser.add_argument('--train_top_n', type=int, default=10)
    parser.add_argument('--downsampling', default=4, type=int)
    return parser.parse_args()


if __name__=='__main__':
    args = parse_args()
    np.random.seed(0)
    # plt.ioff()
    
    print('--------------------------------')
    print('scikit-learn version: %s' % sklearn.__version__)
    print('Note: scikit-learn should be v1.1.2 for external validation.')
    print('--------------------------------')
    
    
    datadir = opj(workdir, '..', 'dataset/')
    tissuedir = opj(workdir, 'results', args.cohort, '3_tissues_32x_downsampled', 'tissue')
    save_dir = opj(workdir, 'results', args.cohort, '4_non_linear_results_32x_downsampled')
    kmeans_save_dir = opj(workdir, 'results', args.cohort, 'IHC_kmeans')
    os.makedirs(kmeans_save_dir, exist_ok=True)
    
    dl_epoch = args.dl_epoch
    k = args.k
    patchsize = 512
    top_m = 10
    
    train_top_n = args.train_top_n
    
    if args.search_k > 0:
        pid_tids = [f for f in os.listdir(save_dir)]
        pid_tids = np.sort(pid_tids)
        
        if not os.path.exists(opj(kmeans_save_dir, 'pv_lab_train_top=%d.pkl' % train_top_n)):
            img_rgb = []
            img_lab = []
            img_lab_resized = []
            for ptid in tqdm(pid_tids):
                patch_dir = opj(save_dir, ptid, 'IHC_nr_patchs')
                if not os.path.exists(patch_dir): continue
                top_patches_ihc = []
                for i in np.arange(train_top_n):
                    if os.path.exists(opj(patch_dir, 'top_%d.png' % (i+1) )):
                        top_patches_ihc.append(opj(patch_dir, 'top_%d.png' % (i+1) ))
                # top_patches_ihc = glob.glob(opj(patch_dir, '*.png'))
                for tpi in top_patches_ihc:
                    img = Image.open(tpi)
                    img_rgb.append(img)
                    img_lab.append(color.rgb2lab(img))
                    newsize = (int(img.size[0]/args.downsampling), int(img.size[1]/args.downsampling))
                    img_resized = np.array(img.resize(newsize))
                    img_lab_resized.append(color.rgb2lab(img_resized))
            img_rgb = np.stack(img_rgb)
            
            with open(opj(kmeans_save_dir, 'img_rgb.pkl'), 'wb') as f:
                pickle.dump(img_rgb, f)
            del img_rgb
            
            img_lab = np.stack(img_lab)
            with open(opj(kmeans_save_dir, 'img_lab.pkl'), 'wb') as f:
                pickle.dump(img_lab, f, protocol=4)
            del img_lab
            img_lab_resized = np.stack(img_lab_resized)
            img_lab_shape = img_lab_resized.shape
            print('Total number of patches: %d' % img_lab_shape[0])
            pv_lab_train = img_lab_resized.reshape(-1,3)
            with open(opj(kmeans_save_dir, 'img_lab_resized.pkl'), 'wb') as f:
                pickle.dump(img_lab_resized, f, protocol=4)
            with open(opj(kmeans_save_dir, 'pv_lab_train_top=%d.pkl' % train_top_n), 'wb') as f:
                pickle.dump(pv_lab_train, f, protocol=4)
        else:
            with open(opj(kmeans_save_dir, 'pv_lab_train_top=%d.pkl' % train_top_n), 'rb') as f:
                pv_lab_train = pickle.load(f)
        
        
        starttime = time.time()
        _, kmeans = search_kmeans(kmeans_save_dir, pv_lab_train, search_k = False, minibatch=args.minibatchKmeans, maxk=args.search_k)
        print('Time elapsed: %.2f s' % (time.time() - starttime))
        
        print('All done. Exit.')


    
    elif args.search_k == 0:
        with open(opj(kmeans_save_dir, 'img_rgb.pkl'), 'rb') as f:
            img_rgb = pickle.load(f)
        with open(opj(kmeans_save_dir, 'img_lab.pkl'), 'rb') as f:
            img_lab = pickle.load(f)
        with open(opj(kmeans_save_dir, 'K=%d_kmeansModel.pkl' % k), 'rb') as f:
            kmeans = pickle.load(f)
        # kmeans = create_kmeans_model(k, minibatch=False, batch_size=None)
        # kmeans.fit(pv_lab_train)
        # kmeans.labels_
        centers = kmeans.cluster_centers_
        colors = distinct_colors[1:k+1]
        # assign labels to pixels (L*a*b*)
        pv_lab_train = img_lab.reshape(-1,3)
        del img_lab
        labels = kmeans.predict(pv_lab_train)
        labels = labels.flatten()
        del pv_lab_train
        # convert all pixels to the color of the centroids
        segmented_image = np.zeros((len(labels),3))
        for lbl in np.arange(k):
            segmented_image[labels==lbl,:] = colors[lbl]
        segmented_image = segmented_image.astype(np.uint8)
        # reshape back to the original image dimension
        shape = (int(len(segmented_image)/patchsize**2),
                   patchsize,
                   patchsize,
                   3)
        segmented_image = segmented_image.reshape(shape)
        labels_image = np.repeat(labels[:, np.newaxis], 3, axis=1).reshape(img_rgb.shape)
        del labels
        
    # =============================================================================
    #         #save distinct colors
    # =============================================================================
        example_imgs_dir = opj(kmeans_save_dir, 'example_imgs_k=%d' % k)
        os.makedirs(example_imgs_dir, exist_ok=True)
        for i in tqdm(range(300)):
            this_img_rgb = img_rgb[i, ...]
            this_img_lbl = labels_image[i, ...]
            this_img_lbl_rgb = copy.deepcopy(this_img_lbl)
            for j in range(k):
                idx = this_img_lbl[..., 0] == j
                this_img_lbl_rgb[idx, 0], this_img_lbl_rgb[idx, 1], this_img_lbl_rgb[idx, 2] = distinct_colors[j+1]
            this_img_lbl_rgb = this_img_lbl_rgb.astype(np.uint8)
            this_merged = np.concatenate([this_img_rgb, this_img_lbl_rgb], axis=1)
            this_merged = Image.fromarray(this_merged)
            this_merged.save(opj(example_imgs_dir, '%d.png') % i)
            
            
        colors_rgba = [(c[0]/256,c[1]/256,c[2]/256) for c in colors]
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        ax.bar(np.arange(k), 1, width=0.8, align='center', color=colors_rgba)
        ax.set_yticks([])
        ax.set_title('clustering centers')
        ax.set_xticks(np.arange(k))
        fig.savefig(opj(kmeans_save_dir, 'clustering_center_colors_k=%d.png' %k),dpi=600)
        # plt.show()
        
    
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
