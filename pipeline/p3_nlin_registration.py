#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:48:56 2020

@author: zhihuan
"""
import cv2
import os,sys,platform
import numpy as np
import pandas as pd
import time
from PIL import Image
import matplotlib.pyplot as plt
from imshowpair import imshowpair
import pickle
import argparse
from tqdm import tqdm
'''
pip install SimpleITK
pip install imshowpair
pip install scikit-image
pip install opencv-python
'''

opj = os.path.join

print("Currently working on %s" % list(platform.uname())[1])

workdir = 'workdir'

sys.path.append(os.path.join(workdir, 'ANHIR_MW')) # AGH UST non-linear registration
sys.path.append(os.path.join(workdir, 'python_script'))

from python_script import get_data
from python_script import registration
from python_script import plots

    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=str, default='TNBC')
    parser.add_argument('--downsample', type=str, default='32x')
    parser.add_argument('--bg_color', type=int, default=243)
    parser.add_argument('--pid_tid', type=str, default='P1_1')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
        
    cohort = args.cohort
    
    np.random.seed(0)
    plt.ioff()



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
    
    resultdir = opj(workdir, 'results', cohort, '4_non_linear_results_%s_downsampled' % args.downsample)
    if not os.path.exists(resultdir): os.makedirs(resultdir)

    
    for j in tqdm(tissue_df.index):
        current_data = tissue_df.iloc[j,:]
        pid, tid = tissue_df.loc[j,['pid','tid']]
        
        if ('%s_%s' % (pid,tid)) != args.pid_tid:
            continue

        print('>>> pid: %s  tid: %s' % (pid, tid))
        
        savedir = opj(resultdir, '%s_%d' % (pid, tid))
        if not os.path.exists(savedir): os.makedirs(savedir)
    
        
        
        if not os.path.exists(opj(savedir, 'registration_dict.pickle')):
            st = time.time()
            HE_tissue, IHC_tissue, properties = get_data.get_tissue_image(args, cohort, pid, tid, patient_df, tissue_df)
            print('Tissue array extracted. Time spent: %.2f' % (time.time()-st))
            
            # image registration
            IHC_tissue_nr, return_dict = registration.AGH(source=HE_tissue, target=IHC_tissue, landmarks=None)
            
            print('-------------------------------------------')
            print('Registration finished. Now saving data.')
                    
            
            if 'rTRE' in return_dict: return_dict['rTRE'].to_csv(opj(savedir, 'rTRE.csv'))
            mpp = float(properties['H&E']['aperio.MPP'])
            
            print('-------------------------------------------')
            print('Saving images: Full_resolution_HE.png, Full_resolution_IHC.png, Full_resolution_IHC_nr.png.')
            Image.fromarray(HE_tissue).save(opj(savedir, 'Full_resolution_HE.png'))
            Image.fromarray(IHC_tissue).save(opj(savedir, 'Full_resolution_IHC.png'))
            Image.fromarray(IHC_tissue_nr).save(opj(savedir, 'Full_resolution_IHC_nr.png'))
            
            print('Done. Now save low resolution and fused image (4x)')
            # save low resolution and fused image
            saved_downsampling_ratio = 4
            new_size = (int(HE_tissue.shape[1]/saved_downsampling_ratio), int(HE_tissue.shape[0]/saved_downsampling_ratio))
            
            HE_tissue_lowres = Image.fromarray(HE_tissue).resize(new_size)
            del HE_tissue
            IHC_tissue_lowres = Image.fromarray(IHC_tissue).resize(new_size)
            del IHC_tissue
            IHC_tissue_nr_lowres = Image.fromarray(IHC_tissue_nr).resize(new_size)
            del IHC_tissue_nr
            
            HE_tissue_lowres.save(opj(savedir, 'Downsample=%dx_HE.png' % saved_downsampling_ratio))
            IHC_tissue_lowres.save(opj(savedir, 'Downsample=%dx_IHC.png' % saved_downsampling_ratio))
            IHC_tissue_nr_lowres.save(opj(savedir, 'Downsample=%dx_IHC_nr.png' % saved_downsampling_ratio))

            HE_tissue_lowres = np.array(HE_tissue_lowres)
            IHC_tissue_lowres = np.array(IHC_tissue_lowres)
            IHC_tissue_nr_lowres = np.array(IHC_tissue_nr_lowres)
            
            fused_before = np.dstack([IHC_tissue_lowres[:,:,0], HE_tissue_lowres[:,:,0], HE_tissue_lowres[:,:,2]])
            Image.fromarray(fused_before).save(opj(savedir, 'Downsample=%dx_fused_before.png' % saved_downsampling_ratio))
            
            
            fused_after = np.dstack([IHC_tissue_nr_lowres[:,:,0], HE_tissue_lowres[:,:,0], HE_tissue_lowres[:,:,2]])
            Image.fromarray(fused_after).save(opj(savedir, 'Downsample=%dx_fused_after.png' % saved_downsampling_ratio))
            
            with open(opj(savedir, 'registration_dict.pickle'), 'wb') as f: pickle.dump(return_dict, f)
    
    
        else:
            pass
        
        print('PID: %s, tissue: %s. All Done.' % (pid, tid))
        print('---------------')
        
    
