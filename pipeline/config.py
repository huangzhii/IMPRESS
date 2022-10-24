#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 17:08:53 2020

@author: zhihuang
"""

import platform

def config_wordir():
    print("Currently working on %s" % list(platform.uname())[1])
    workdir = 'Your work directory'
    datadir = {}
    datadir['HER2+'] = 'HER2+ data directory'
    datadir['TNBC'] = 'TNBC data directory'
    
    tissuedir = {}
    tissuedir['HER2+'] = workdir + 'results/HER2+/3_tissues_32x_downsampled/tissueMask/'
    tissuedir['TNBC'] = workdir + 'results/TNBC/3_tissues_16x_downsampled/tissueMask/'
    
    landmarkdir = {}
    landmarkdir['HER2+'] = workdir + 'landmarks/HER2+/'
    landmarkdir['TNBC'] = workdir + 'landmarks/TNBC/'
        
    return workdir, datadir, tissuedir, landmarkdir
