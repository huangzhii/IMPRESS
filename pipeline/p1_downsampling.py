
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 12:57:21 2022

@author: zhihuang
"""

import os, sys, platform
import openslide
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

if __name__ == '__main__':
    resultdir = 'results/'
    datadir = 'dataset/'
    slidelist = os.listdir(datadir)
    
    df = pd.DataFrame(index=np.arange(len(slidelist)),
                      columns=['cohort','pid','type','path'])
    for i, s in enumerate(slidelist):
        cohort = s.split('-')[0]
        pid = s.split('-')[1].split(' ')[0]
        type_ = s.split('(')[1].split(')')[0]
        path = os.path.join(datadir, s)
        df.loc[i,:] = [cohort, pid, type_, path]
        
        
    
    
    for i in tqdm(df.index):
        slide = openslide.OpenSlide(df.loc[i,'path'])
        ld = slide.level_downsamples
        print(slide.properties['aperio.AppMag'])
        if df.loc[i,'cohort'] == 'HER2':
            savedir = os.path.join(resultdir, 'HER2+')
        else:
            savedir = os.path.join(resultdir, 'TNBC')
        os.makedirs(savedir, exist_ok = True)
        os.makedirs(os.path.join(savedir, '1_4x_downsampled'), exist_ok = True)
        os.makedirs(os.path.join(savedir, '1_8x_downsampled'), exist_ok = True)
        os.makedirs(os.path.join(savedir, '1_32x_downsampled'), exist_ok = True)
        fname = df.loc[i,'pid'] +'_'+ df.loc[i,'type'] + '.png'
        if os.path.exists(os.path.join(savedir, '1_32x_downsampled', fname)):
            continue
            
        dimension_4x_down = slide.level_dimensions[1] # 4x downsample (10x)
        img_4x = slide.read_region((0,0), 1, dimension_4x_down)
        img_8x = img_4x.resize((int(np.round(img_4x.size[0]/2)), int(np.round(img_4x.size[1]/2))), resample=Image.BICUBIC)
        img_32x = img_4x.resize((int(np.round(img_4x.size[0]/8)), int(np.round(img_4x.size[1]/8))), resample=Image.BICUBIC)

        img_4x.save(os.path.join(savedir, '1_4x_downsampled', fname))
        img_8x.save(os.path.join(savedir, '1_8x_downsampled', fname))
        img_32x.save(os.path.join(savedir, '1_32x_downsampled', fname))



        
        
