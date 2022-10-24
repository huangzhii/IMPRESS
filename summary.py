#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import copy
import warnings
import glob
from sklearn.metrics import roc_curve, auc
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=str, default='HER2+')
    return parser.parse_args()

def index2label(multiindex):
    labels = []
    for i1, i2 in multiindex:
        lbl1 = i1.capitalize()
        if lbl1 == 'Lymphocytes':
            lbl1 = 'Lymph'
        lbl2 = i2.replace('_', ' ')
        if 'PDL1' in lbl2:
            lbl2 = lbl2.replace('PDL1', 'PD-L1')
        if 'PD1' in lbl2:
            lbl2 = lbl2.replace('PD1', 'PD-1')
        if 'HE' in lbl2 and 'HER2' not in lbl2:
            lbl2 = lbl2.replace('HE', 'H&E')
        labels.append(lbl1+': '+lbl2)
    return labels


def plot_coefficients(args, fig, axes, coef, title=None):
    
    df = copy.deepcopy(coef)
    df = df.sort_values(by=['mean','std'], ascending = True)
    axes.grid(True, linestyle='dotted')
    axes.set_axisbelow(True)
    labels = index2label(df.index)
    ftype = np.array([l.split(':')[0] for l in labels])
    colormap = {'Clinical':'seagreen',
                'Stroma':'pink',
                'Tumor':'violet',
                'Lymph':'dodgerblue',
                'All':'salmon'}
    if 'pathologists' in coef.index.get_level_values(0):
        colormap['Pathologists'] = 'grey'
    colors = []
    for t in ftype:
        colors.append(colormap[t])
    
    values = df.loc[:,'mean'].values.astype(float)
    std = df.loc[:,'std'].values.astype(float)
    rects = axes.barh(labels, values, xerr=std, align='center', color=colors)
    axes.tick_params(axis='x', labelrotation=90)
    axes.set_xlabel('coefficient',fontsize=12)
    handles = [plt.Rectangle((0,0),1,1, color=colormap[c]) for c in colormap]
    fig.legend(handles, colormap.keys(), loc='lower right', bbox_to_anchor=(0.95, 0.12))
    if title is not None:
        fig.suptitle(t=title, fontsize=12)
    return fig

def plot_ROC_curves(fig, ax, y_all, perf, title=None):
    """
    Plots the ROC curves and calculate AUC
    """
    curves = {'IMPRESS_all': 'royalblue',
              'IMPRESS_HE_only': 'plum',
              'IMPRESS_IHC_only': 'pink',
              'pathologists_eval': 'tomato'}
    
    type_convert = {'IMPRESS_all': 'IMPRESS',
                      'IMPRESS_HE_only': 'IMPRESS (H&E only)',
                      'IMPRESS_IHC_only': 'IMPRESS (IHC only)',
                      'pathologists_eval': 'Pathologists'}
    
    for fgroup in curves.keys():
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        ax.set_aspect('equal')
        for seed in range(int(y_all[fgroup].shape[1]/3)):
            y_true = y_all[fgroup].loc[:,'y_true'].iloc[:,seed]
            y_pred_proba = y_all[fgroup].loc[:,'y_pred_proba'].iloc[:,seed]
            tpr, fpr, treshold = roc_curve(y_true, 1-y_pred_proba)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            ax.plot(fpr, tpr, color=curves[fgroup], linewidth=2, alpha=0.10, label=None)
    
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
    
        ax.plot(mean_fpr, mean_tpr, color=curves[fgroup],
                  label=r'%s (AUC = %0.4f $\pm$ %0.2f)' % \
                      (type_convert[fgroup], perf[fgroup].loc['AUC','mean'], perf[fgroup].loc['AUC','std']),
                  linewidth=3.0, alpha=0.80)
        
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        
        if fgroup == 'IMPRESS_all':
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.1,
                             label=r'$\pm$ 1 standard deviation')
        else:
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.1,
                             label=None)
    
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    x = [0.0, 1.0]
    plt.plot(x, x, linestyle='dashed', color='red', linewidth=2.0, label='Random')
    plt.legend(fontsize=10, loc='best')
    
    if title is not None:
        fig.suptitle(t=title, fontsize=12)
    return fig


if __name__ == '__main__':
    plt.rcParams["font.family"] = "Myriad Pro"
    plt.ioff()
    args = parse_args()
    cohort = args.cohort
    writedir_root = os.path.join('results', cohort)
    
    groups = ['IMPRESS_all', 'IMPRESS_HE_only', 'IMPRESS_IHC_only', 'pathologists_eval']
    y_dict = {}
    perf_dict = {}
    performance_group = {}
    for fgroup in groups:
        writedir = os.path.join(writedir_root, fgroup)
        
        coefficients_all = pd.DataFrame()
        performance_all = pd.DataFrame()
        y_all = pd.DataFrame()
        for sdir in glob.glob(os.path.join(writedir, 'seed=*')):
            coefficients = pd.read_csv(os.path.join(sdir, 'coefficients.csv'), index_col=[0,1])
            coefficients_all = pd.concat((coefficients_all, coefficients), axis=1)
            performance = pd.read_csv(os.path.join(sdir, 'performance.csv'), index_col=0)
            performance_all = pd.concat((performance_all, performance.T), axis=1)
            y = pd.read_csv(os.path.join(sdir, 'y_all.csv'), index_col=0)
            y_all = pd.concat((y_all, y), axis=1)
    
        coef_mean = coefficients_all.mean(axis=1)
        coef_std = coefficients_all.std(axis=1)
        coef = pd.concat((coef_mean, coef_std), axis=1)
        coef.columns = ['mean','std']
        coef = coef.sort_values(by='mean', ascending = False)
        coef.to_csv(os.path.join(writedir, 'coeffcient_mean_std.csv'))
        performance_group[fgroup] = performance_all
    
        perf_mean = performance_all.mean(axis=1)
        perf_std = performance_all.std(axis=1)
        perf = pd.concat((perf_mean, perf_std), axis=1)
        perf.columns = ['mean','std']
        perf.to_csv(os.path.join(writedir, 'performance_mean_std.csv'))
            
        coef_short = coef
        fig, axes = plt.subplots(figsize=(5,8), constrained_layout=True)
        fig = plot_coefficients(args, fig, axes, coef_short, title = cohort)
        fig.savefig(os.path.join(writedir, 'coefficients.pdf'), dpi=600)
        y_dict[fgroup] = y_all
        perf_dict[fgroup] = perf
        
        print('%s mean AUC: %.8f' % (fgroup, perf_mean['AUC']))
    
    
    # plot roc curve
    fig, ax = plt.subplots(figsize=(5,5), constrained_layout=True)
    fig = plot_ROC_curves(fig, ax, y_dict, perf_dict, title = cohort + ' cohort ROC curve')
    fig.savefig(os.path.join(writedir, '..', 'ROC_curve.pdf'), dpi=600)

    # =============================================================================
    # t-test
    # =============================================================================
    from scipy.stats import ttest_ind
    '''
    This is a two-sided test for the null hypothesis that 2 independent samples
    have identical average (expected) values.
    This test assumes that the populations have identical variances by default.
    '''
    fgroup_auc = pd.DataFrame(columns = groups)
    for fgroup in groups:
        fgroup_auc[fgroup] = performance_group[fgroup].loc['AUC',:].values

    t_test = pd.DataFrame(index = groups, columns = groups)
    for g1 in groups:
        for g2 in groups:
            s, p = ttest_ind(fgroup_auc[g1], fgroup_auc[g2])
            t_test.loc[g1,g2] = 't=%.4f (p=%.4E)' % (s,p)
    t_test.to_csv(os.path.join(writedir, '..', 't_test_AUC.csv'))








    
    