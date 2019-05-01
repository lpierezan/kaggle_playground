#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 18:27:05 2019

Utilitary classes and functions

@author: lucas
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR, NuSVR
from sklearn.linear_model import LinearRegression, ElasticNet
from scipy import stats
import datetime


def grid_plot(df, y, cols = None, plot_func = sns.boxplot, n_row = 3, figsize = (25,30)):
    if cols is None:
        cols = df.columns
        
    shape = (int(np.ceil(len(cols)/n_row)), n_row)
    fig, axs = plt.subplots(shape[0],shape[1], figsize = figsize)
    
    for i,c in enumerate(cols):
        plot_func(x = c, y = y, data = df, ax = axs[i//n_row, i%n_row] if shape[0] > 1 else axs[i%n_row])
        
def df_info(df):
    features_info = pd.DataFrame(data = {'dtype' : df.dtypes, 
                             'nan_count' : df.isnull().sum(axis = 0),
                             '0_count' : (df == 0).sum(axis = 0),
                             'unique_count': df.nunique(dropna=False)
                            })
    return features_info


def fv_info(series, value, y, base_loss, loss):
    dic_ret = {}
    if pd.isnull(value):
        mask = series.isnull()
    else:
        mask = (series == value)

    n = mask.sum()
    dic_ret['feature'] = series.name
    dic_ret['value'] = str(value)
    dic_ret['count'] = n
    dic_ret['p'] = dic_ret['count']/len(series)
    dic_ret['y_in_mean'] = y[mask].mean()
    dic_ret['loss_in'] = loss(y[mask], dic_ret['y_in_mean']) if n > 0 else 0
    dic_ret['loss_out'] = loss(y[~mask], y[~mask].mean())
    dic_ret['loss'] = (dic_ret['p']*dic_ret['loss_in']**2 + (1 - dic_ret['p'])*dic_ret['loss_out']**2)**0.5
    dic_ret['delta_loss'] = base_loss - dic_ret['loss']
    
    return dic_ret


def split_info(df_imp, y, cols = 'all', loss = None):
    if loss is None:
        loss = lambda y_true, y_pred : mean_squared_error(y_true, y_pred + np.zeros_like(y_true))**0.5
    
    base_loss = loss(y, y.mean())
    
    if isinstance(cols, str) and cols == 'all':
        cols = list(df_imp.columns)
        
    dfs = []
    for col in cols:
        series = df_imp[col]
        values = list(set(series[~series.isnull()]))
        values += [None] if series.isnull().sum() > 0 else []
        order = ['feature', 'value', 'count', 'p', 'y_in_mean','loss_in', 'loss_out', 'loss', 'delta_loss']
        records = list(map(lambda value: fv_info(series, value, y, base_loss, loss), values))
        df_col = pd.DataFrame(records, columns=order)
        dfs.append(df_col)
        
    return pd.concat(dfs, axis = 0, ignore_index=True).sort_values('delta_loss', ascending = False)