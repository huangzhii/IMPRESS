#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 12:32:26 2020

@author: zhihuang
"""
import os
import numpy as np
from PIL import Image

def save_registration_patches(source, target, source_landmark, target_landmark, resultdir, size):
    
    patchfolderdir = '%sregistered_patches_size=%d/' % (resultdir, size)
    if not os.path.exists(patchfolderdir): os.mkdir(patchfolderdir)
    
    source = Image.fromarray(source)
    target = Image.fromarray(target)
    
    gap = int(round(size/100))
    big_image_array = np.zeros((len(source_landmark)*(size+gap)-gap, (size+gap)*3-gap,3))
    for i in range(len(source_landmark)):
        center = source_landmark[i,:]
        x = max(0, int(round(center[0]-size/2)))
        y = max(0, int(round(center[1]-size/2)))
        sp = source.crop((x, y, x+size, y+size))
        
        center = target_landmark[i,:]
        x = max(0, int(round(center[0]-size/2)))
        y = max(0, int(round(center[1]-size/2)))
        tp = target.crop((x, y, x+size, y+size))
        
        sp.save(patchfolderdir + '%03d_he.png' % (i+1))
        tp.save(patchfolderdir + '%03d_ihc.png' % (i+1))
        
        fused = np.dstack([np.array(tp)[:,:,0], np.array(sp)[:,:,0], np.array(sp)[:,:,2]])
        
        big_image_array[i*(size+gap):(i*(size+gap)+size),0:size,:] = np.array(sp)
        big_image_array[i*(size+gap):(i*(size+gap)+size),(size+gap):(size*2+gap),:] = np.array(tp)
        big_image_array[i*(size+gap):(i*(size+gap)+size),(size+gap)*2:((size+gap)*3-gap),:] = fused

    Image.fromarray(np.uint8(big_image_array)).save(patchfolderdir + 'all.png')
    
    
    
def save_HE_seg_patches(source, segmentation, source_landmark, resultdir, size):
    
    patchfolderdir = '%sHE_seg_patches_size=%d/' % (resultdir, size)
    if not os.path.exists(patchfolderdir): os.mkdir(patchfolderdir)
    
    source = Image.fromarray(source)
    segmentation = Image.fromarray(segmentation)
    
    gap = int(round(size/100))
    big_image_array = np.zeros((len(source_landmark)*(size+gap)-gap, (size+gap)*3-gap,3))
    for i in range(len(source_landmark)):
        center = source_landmark[i,:]
        x = max(0, int(round(center[0]-size/2)))
        y = max(0, int(round(center[1]-size/2)))
        sp = source.crop((x, y, x+size, y+size))
        seg = segmentation.crop((x, y, x+size, y+size))
        
        
        sp.save(patchfolderdir + '%03d_he.png' % (i+1))
        seg.save(patchfolderdir + '%03d_he_seg.png' % (i+1))
        
        fused = np.uint8(np.array(sp)/2 + np.array(seg)/2)
        
        big_image_array[i*(size+gap):(i*(size+gap)+size),0:size,:] = np.array(sp)
        big_image_array[i*(size+gap):(i*(size+gap)+size),(size+gap):(size*2+gap),:] = np.array(seg)
        big_image_array[i*(size+gap):(i*(size+gap)+size),(size+gap)*2:((size+gap)*3-gap),:] = fused

    Image.fromarray(np.uint8(big_image_array)).save(patchfolderdir + 'all.png')
    
def save_IHC_seg_patches(source, segmentation, source_landmark, resultdir, size):
    
    patchfolderdir = '%sIHC_nr_seg_patches_size=%d/' % (resultdir, size)
    if not os.path.exists(patchfolderdir): os.mkdir(patchfolderdir)
    
    source = Image.fromarray(source)
    segmentation = Image.fromarray(segmentation)
    
    gap = int(round(size/100))
    big_image_array = np.zeros((len(source_landmark)*(size+gap)-gap, (size+gap)*3-gap,3))
    for i in range(len(source_landmark)):
        center = source_landmark[i,:]
        x = max(0, int(round(center[0]-size/2)))
        y = max(0, int(round(center[1]-size/2)))
        sp = source.crop((x, y, x+size, y+size))
        seg = segmentation.crop((x, y, x+size, y+size))
        
        
        sp.save(patchfolderdir + '%03d_ihc_nr.png' % (i+1))
        seg.save(patchfolderdir + '%03d_ihc_nr_seg.png' % (i+1))
        
        fused = np.uint8(np.array(sp)/2 + np.array(seg)/2)
        
        big_image_array[i*(size+gap):(i*(size+gap)+size),0:size,:] = np.array(sp)
        big_image_array[i*(size+gap):(i*(size+gap)+size),(size+gap):(size*2+gap),:] = np.array(seg)
        big_image_array[i*(size+gap):(i*(size+gap)+size),(size+gap)*2:((size+gap)*3-gap),:] = fused

    Image.fromarray(np.uint8(big_image_array)).save(patchfolderdir + 'all.png')
    
    
    
    
    
    
    
    
