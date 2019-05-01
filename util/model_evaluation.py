#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 19:17:35 2019

@author: lpierezan
"""

from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

def evaluate_model(model, df, y, cv = 5):
    kfold = KFold(n_splits = cv, shuffle=True, random_state=42)
    scores = cross_validate(model, df, y, scoring='neg_mean_squared_error', return_train_score=True, cv = kfold)
    
    print('RMSE train: ', (-scores['train_score'].mean())**0.5, ' +-', scores['train_score'].std()**0.5)
    print('RMSE test: ', (-scores['test_score'].mean())**0.5, ' +-', scores['test_score'].std()**0.5)
    model.fit(df,y)
    print('RMSE train:', mean_squared_error(y, model.predict(df))**0.5)
    
    if hasattr(model, 'feature_importances_'):
        fig , ax = plt.subplots(figsize = (10,10))
        sns.barplot(x = 'importance', y = 'feature', 
                    data = pd.DataFrame({'feature' : df.columns, 'importance' : model.feature_importances_}).\
                        sort_values('importance', ascending = False), ax = ax)
    
    if hasattr(model, 'train_score_'):
        plt.show()
        plt.plot(model.train_score_)
    
    return model

kfold = KFold(n_splits=5, shuffle=True, random_state=42)