#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import requests
import os
# settings to display all columns
pd.set_option("display.max_columns", None)


# In[2]:


DATIUM_DATASET_URL = 'https://drive.google.com/uc?export=download&id=1kXe7p6o6IDr9KwfOG2vPk8TEDsnMukAt'
DATIUM_DATASET_PATH = '../datasets/DatiumSample.rpt'
OUTPUT_DATASET_PATH = '../datasets/dataset_1.csv'
DROP_COLS = ['AvgWholesale', 'AvgRetail', 'GoodWholesale', 'GoodRetail', 'TradeMin', 'TradeMax', 'PrivateMax']
TARGET_COL = 'Sold_Amount'
CUTOFF_NULL_PCT = 25 # Maximum % of Nulls Before Dropping Column


# In[3]:


if not os.path.exists(DATIUM_DATASET_PATH):
    print(f'INFO - Downloading Datium Dataset to {DATIUM_DATASET_PATH}')
    response=requests.get(DATIUM_DATASET_URL)
    with open('../datasets/DatiumSample.rpt', 'wb')as file:
        file.write(response.content)
else:
    print(f'INFO - Skip Download. Datium Dataset found at {DATIUM_DATASET_PATH}')


# ## Read Data

# In[4]:


df=pd.read_csv(DATIUM_DATASET_PATH,sep='\t')
df=df.drop(DROP_COLS, axis=1)
nrows, ncols = df.shape
print(f'INFO - Dataset contains {nrows} rows, {ncols} columns')
df.head()


# ## Remove Rows with Null Targets, New Price, KM

# In[5]:


def remove_null_rows(df, col):
    target_nulls = df[col].isna()
    if target_nulls.sum():
        print(f'WARNING - Found {target_nulls.sum()} Null Values in : {col}. Will drop rows')
        df = df[~target_nulls]
        nrows, ncols = df.shape
        print(f'INFO - Dataset contains {nrows} rows, {ncols} columns')   
    return df


df=remove_null_rows(df, TARGET_COL)
df=remove_null_rows(df, 'NewPrice')


# ## Check Columns with Mixed D_Types

# In[6]:


check_cols=[27, 91]
for c in check_cols:
    check_col=df.columns[c]
    print(check_col)
    print('D_Types are: ',set(type(s) for s in df[check_col]))


# In[7]:


print('INFO - EngineDescription is just a rounded up number for EngineSize')
df[['EngineDescription','EngineSize']]
## ***Add Round Up Verification


# In[8]:


print('INFO - Drop EngineDescription')
df=df.drop('EngineDescription', axis=1)
nrows, ncols = df.shape
print(f'INFO - Dataset contains {nrows} rows, {ncols} columns')


# In[9]:


nulls_pct = df['NormalChargeVoltage'].isna().sum()/nrows*100
print(f'INFO - {round(nulls_pct,3)}% Nulls in NormalChargeVoltage')
print('INFO - Drop NormalChargeVoltage')

df=df.drop('NormalChargeVoltage', axis=1)
nrows, ncols = df.shape
print(f'INFO - Dataset contains {nrows} rows, {ncols} columns')


# ## Drop Columns with many Nulls

# In[10]:


null_cols=df.isnull().sum(axis=0)
null_cols=null_cols[null_cols>0]


# In[11]:


print(f'INFO - Dropping Columns with Nulls Rows Larger than Cutoff {CUTOFF_NULL_PCT}%')
drop_cols = null_cols[null_cols> nrows * CUTOFF_NULL_PCT / 100]
print(f'INFO - Dropping {len(drop_cols)}/{len(df.columns)} Numerical Columns due to Nulls')
print(drop_cols)


# # End of Preprocess 1

# In[12]:


df.to_csv(OUTPUT_DATASET_PATH, index=None)
print(f'INFO - End of Preprocess 1. Saved dataset to {OUTPUT_DATASET_PATH}')

