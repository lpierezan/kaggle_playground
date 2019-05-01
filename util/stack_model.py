#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 19:17:55 2019

@author: lpierezan
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.base import BaseEstimator
from sklearn.base import clone
import pandas as pd
import numpy as np

class stack_model(BaseEstimator):
    
    def __init__(self, cols1 = None, cols2 = None):
        self.rf_stack = RandomForestRegressor(n_estimators=100, max_features=None, max_depth = 5, min_impurity_decrease = 0.0,  min_samples_split = 10, min_samples_leaf=10, bootstrap=True, random_state=42)
        self.lm_stack = Lasso(alpha=0.001, normalize=True, max_iter=1000, random_state=42)
        self.cols1 = cols1
        self.cols2 = cols2
        self.lm_models = {}
    
    def get_params(self, deep=True):
        return {'cols1' : self.cols1, 'cols2' : self.cols2}

    def set_params(self, **parameters):
        self.cols1 = parameters['cols1']
        self.cols2 = parameters['cols2']        
        return self

    def fit(self, df, y):
        
        cols1 = list(df.columns) if self.cols1 is None else self.cols1
        cols2 = list(df.columns) if self.cols2 is None else self.cols2
            
        self.rf_stack.fit(df[cols1],y)
        leaf = self.rf_stack.apply(df[cols1])
        
        #one lm model for every rf estimator and leaf
        for f_idx in range(leaf.shape[1]):
            for leaf_num, idxs in pd.DataFrame(leaf[:,f_idx]).reset_index().groupby(0):
                idxs = idxs['index'].values
                df_leaf = df[cols2].iloc[idxs].copy()
                y_leaf = y.iloc[idxs].copy()
                lm_model = clone(self.lm_stack)
                lm_model.fit(df_leaf,y_leaf)
                self.lm_models[(f_idx, leaf_num)] = lm_model
                
        return self
    
    def predict(self, df):
        
        cols1 = list(df.columns) if self.cols1 is None else self.cols1
        cols2 = list(df.columns) if self.cols2 is None else self.cols2

        leaf = self.rf_stack.apply(df[cols1])
        stack_preds = np.zeros_like(leaf, dtype=float)
        
        #predict unsing lm models for every rf estimator and leaf
        for f_idx in range(leaf.shape[1]):
            for leaf_num, idxs in pd.DataFrame(leaf[:,f_idx]).reset_index().groupby(0):
                idxs = idxs['index'].values
                df_leaf = df[cols2].iloc[idxs].copy()
                lm_model = self.lm_models[(f_idx, leaf_num)]
                leaf_pred = lm_model.predict(df_leaf)
                stack_preds[idxs,f_idx] = leaf_pred
        
        y_pred = stack_preds.mean(axis = 1)
        return y_pred
        
#check_estimator(stack_model)