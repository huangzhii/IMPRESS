#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 19:31:09 2020

@author: zhihuang
"""
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage import color
from ANHIR_MW import *
import ANHIR_MW.anhir_method as am
import ANHIR_MW.utils as utils

def AGH(source, target, landmarks):
    source_grey = color.rgb2gray(source)
    target_grey = color.rgb2gray(target)
    
    p_source, p_target, ia_source, ng_source, nr_source, i_u_x, i_u_y, u_x_nr, u_y_nr, \
        warp_resampled_landmarks, warp_original_landmarks, return_dict = am.anhir_method(target_grey, source_grey)

    u_x_nr_full_res, u_y_nr_full_res = utils.resample_displacement_field(u_x_nr, u_y_nr, source_grey.shape[1], source_grey.shape[0])

    target_nr_r = utils.warp_image(target[:,:,0], u_x_nr_full_res, u_y_nr_full_res)
    target_nr_g = utils.warp_image(target[:,:,1], u_x_nr_full_res, u_y_nr_full_res)
    target_nr_b = utils.warp_image(target[:,:,2], u_x_nr_full_res, u_y_nr_full_res)
    
    target_nr = np.dstack([target_nr_r, target_nr_g, target_nr_b]).astype(float)
    del target_nr_r, target_nr_g, target_nr_b
    target_nr = (target_nr - np.min(target_nr))*255 / (np.max(target_nr) - np.min(target_nr))
    target_nr = target_nr.astype(np.uint8)
    
    
    if landmarks is not None:
        source_landmarks = landmarks.loc[:,['X_HE','Y_HE']].values
        target_landmarks = landmarks.loc[:,['X_IHC','Y_IHC']].values
        
        transformed_source_landmarks = warp_original_landmarks(source_landmarks)
        transformed_target_landmarks = target_landmarks - (transformed_source_landmarks - source_landmarks)
    
        resampled_source_landmarks, transformed_resampled_source_landmarks, resampled_target_landmarks = warp_resampled_landmarks(source_landmarks, target_landmarks)
    
        y_size, x_size, _ = np.shape(target)
        rTRE = pd.DataFrame(index = ['mean', 'median', 'max', 'min'], \
                            columns = ['Initial Original', 'Transformed Original', 'Initial Resampled', 'Transformed Resampled'])
        print("Initial original rTRE: ")
        rTRE.loc[:,'Initial Original'] = utils.print_rtre(source_landmarks, target_landmarks, x_size, y_size)
        print("Transformed original rTRE: ")
        rTRE.loc[:,'Transformed Original'] = utils.print_rtre(transformed_source_landmarks, target_landmarks, x_size, y_size)
    
        y_size, x_size = np.shape(p_target)
        print("Initial resampled rTRE: ")
        rTRE.loc[:,'Initial Resampled'] = utils.print_rtre(resampled_source_landmarks, resampled_target_landmarks, x_size, y_size)
        print("Transformed resampled rTRE: ")
        rTRE.loc[:,'Transformed Resampled'] = utils.print_rtre(transformed_resampled_source_landmarks, resampled_target_landmarks, x_size, y_size)
        
        return_dict['rTRE'] = rTRE
        return_dict['transformed_source_landmarks'] = transformed_source_landmarks
        return_dict['transformed_target_landmarks'] = transformed_target_landmarks
        return_dict['resampled_source_landmarks'] = resampled_source_landmarks
        return_dict['transformed_resampled_source_landmarks'] = transformed_resampled_source_landmarks
        return_dict['resampled_target_landmarks'] = resampled_target_landmarks
        
    
    
    return target_nr, return_dict
