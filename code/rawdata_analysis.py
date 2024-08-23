#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 00:27:18 2024

@author: menglinghe
"""

import os
import pandas as pd
import numpy as np

os.getcwd()
os.chdir('/Users/menglinghe/Library/Mobile Documents/com~apple~CloudDocs/UTK/GRA-UTK/UTK-medical/code')



###### the original dataset without datacleaning
PATH = '../data/'
FILES = ['UTK_UTMC_OBGYN_DEIDENTIFIED_2023-4-26_Data.xlsx', 'UTK_UTMC_OBGYN_DEIDENTIFIED_2023-4-26_VariableNames.xlsx']
df0 = pd.read_excel(PATH+FILES[0], index_col=0)
var = pd.read_excel(PATH+FILES[1])
df0.shape

df3 = pd.read_excel(PATH+FILES[0], index_col=0,dtype={'C': str})

pd.read_excel(file_path, )


var['Type'].value_counts()
var.loc[var['Type']=='Text']


##### calculate missing percentage
columnnames=df0.columns
data_type=df0.dtypes
missing_percentage=df0.isnull().mean().round(4).mul(100)

features= pd.concat([data_type,missing_percentage],axis=1)
features = features.reset_index()
features.columns = ['variable', 'data type', 'missing percentage']

features.to_csv('../data/featues_missing.csv')

##### variables with 75% missingness were analyzed
v_bigmissing = features['variable'][features['missing percentage'] > 75]
features_keep = features['variable'][features['missing percentage'] < 75]



#####  remove high missing percentage missing variables----- leave it after data preprocessing