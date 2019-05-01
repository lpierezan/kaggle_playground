#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 19:18:09 2019

@author: lpierezan
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer


class T:
    
    def __init__(self, bin_th = 0.0, numeric_imputer = None, log_eps = 0.1):
        self.bin_th = bin_th
        self.log_eps = log_eps
        if numeric_imputer is None:
            self.numeric_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        else:
            self.numeric_imputer = numeric_imputer
            
        
    def _fit_regular_cat(self, df, regular_cat = []):
        # imput na + ohe
        self.regular_cat = regular_cat 
        cat_imputer_None = SimpleImputer(missing_values = None, strategy='constant', fill_value='NAN')
        cat_imputer_nan = SimpleImputer(missing_values = np.nan, strategy='constant', fill_value='NAN')
        cat_ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        
        self.regular_cat_pipe = None
        if len(self.regular_cat) > 0:
            self.regular_cat_pipe = Pipeline([('cat_imputer_None',cat_imputer_None),
                                              ('cat_imputer_nan',cat_imputer_nan),
                                              ('cat_ohe', cat_ohe)])
            
            self.regular_cat_pipe.fit(df[self.regular_cat])
            self.regular_cat_tcols = self.regular_cat_pipe.steps[-1][1].get_feature_names(self.regular_cat)
    
    def _fit_to_kbin(self, df):
        # tuples with (column_name, n_bins, strategy)
        
        self.kbin_fit_dic = {}
        
        if len(self.to_kbin) == 0:
            return
            
        for col, n_bins, strg in self.to_kbin:
            kb = KBinsDiscretizer(n_bins=n_bins, strategy=strg, encode='onehot-dense')            
            kb.fit(df[[col]])
            
            #todo improve col names
            kb_cols = ['{}_{}{}_{}{}'.format(col, '<' if i == 0 else '', y1, 
                                             y2, '>' if i == (len(kb.bin_edges_[0])-2) else '')
                       for i,(y1,y2) in enumerate(zip(kb.bin_edges_[0] , kb.bin_edges_[0][1:]))]
            
            self.kbin_fit_dic[col] =  kb, kb_cols
        
    
    def fit(self, df, to_keep = [], to_log = [], to_bin = [], regular_cat = [], to_kbin = [], to_num = [], multiple_col_cat = []):
        self.to_keep = to_keep #stateless
        self.to_log = to_log #stateless
        self.to_bin = to_bin #stateless
        self.to_num = to_num #stateless
        self.to_kbin = to_kbin
        
        
        # every numeric col is imputed with mode
        self.numeric_cols = self.to_keep + self.to_log + self.to_bin + \
        [v[0] for v in self.to_kbin] + [v[0] for v in self.to_num]
        
        df = df.copy()
        # change to numeric
        for c, d in self.to_num:
            df[c] = (df[c].apply(lambda v : d[v] if v in d else np.nan)).astype(float)
        
        if len(self.numeric_cols) > 0:
            self.numeric_imputer.fit(df[self.numeric_cols])
        
        # numerical kbin (todo : use kbin after numeric imputation?)
        self._fit_to_kbin(df)
        
        # categoricals
        self._fit_regular_cat(df, regular_cat)
        
        # todo
        self.multiple_col_cat = multiple_col_cat
        
        
    def transform(self, df):        
        df_ret = pd.DataFrame(index = df.index)
        df = df.copy()
        
        # change to numeric
        for c, d in self.to_num:
            df[c] = (df[c].apply(lambda v : d[v] if v in d else np.nan)).astype(float)
        
        if len(self.numeric_cols) > 0:
            # impute data for numeric cols
            numeric_cols = self.numeric_cols
            df_num = pd.DataFrame(self.numeric_imputer.transform(df[numeric_cols]) , index = df.index, columns=numeric_cols, dtype=float)

            # apply numeric transformers
            df_ret[self.to_keep +[v[0] for v in self.to_num]] = df_num[self.to_keep+[v[0] for v in self.to_num]]
            df_ret[self.to_log] = np.log(df_num[self.to_log] + self.log_eps)
            df_ret[self.to_bin] = (df_num[self.to_bin] > self.bin_th).astype(int)
        
        #to kbin
        if len(self.to_kbin) > 0:
            dfs_kbin = []
            for col, (kb_fitted,kb_cols) in self.kbin_fit_dic.items():
                dfs_kbin.append(pd.DataFrame(kb_fitted.transform(df_num[[col]]), columns=kb_cols))
            
            df_ret = pd.concat([df_ret] + dfs_kbin, axis = 1, verify_integrity=True)
            
        #regular cat
        if len(self.regular_cat) > 0:
            df_reg_cat = pd.DataFrame(data = self.regular_cat_pipe.transform(df[self.regular_cat]), columns = self.regular_cat_tcols)
            df_ret = pd.concat([df_ret, df_reg_cat], axis = 1, verify_integrity=True)
            
        
            
        return df_ret
    
