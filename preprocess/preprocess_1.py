#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import requests
import os
# settings to display all columns
pd.set_option("display.max_columns", None)


# In[20]:


DATIUM_DATASET_URL = 'https://drive.google.com/uc?export=download&id=1kXe7p6o6IDr9KwfOG2vPk8TEDsnMukAt'
DATIUM_DATASET_PATH = '../datasets/DatiumSample.rpt'
OUTPUT_DATASET_PATH = '../datasets/dataset_1.csv'
DROP_COLS = ['AvgWholesale', 'AvgRetail', 'GoodWholesale', 'GoodRetail', 'TradeMin', 'TradeMax', 'PrivateMax']
TARGET_COL = 'Sold_Amount'
CUTOFF_NULL_PCT = 10 # Maximum % of Nulls Before Dropping Column


# In[18]:


if not os.path.exists(DATIUM_DATASET_PATH):
    print(f'INFO - Downloading Datium Dataset to {DATIUM_DATASET_PATH}')
    response=requests.get(DATIUM_DATASET_URL)
    with open('../datasets/DatiumSample.rpt', 'wb')as file:
        file.write(response.content)
else:
    print(f'INFO - Skip Download. Datium Dataset found at {DATIUM_DATASET_PATH}')


# ## Read Data

# In[ ]:


df=pd.read_csv(DATIUM_DATASET_PATH,sep='\t')
df=df.drop(DROP_COLS, axis=1)
nrows, ncols = df.shape
print(f'INFO - Dataset contains {nrows} rows, {ncols} columns')
df.head()


# ## Remove Rows with Null Targets, New Price, KM

# In[ ]:


def remove_null_rows(df, col):
    target_nulls = df[col].isna()
    if target_nulls.sum():
        print(f'WARNING - Found {target_nulls.sum()} Null Values in : {col}. Will drop rows')
        df = df[~target_nulls]
        nrows, ncols = df.shape
        print(f'INFO - Dataset contains {nrows} rows, {ncols} columns')   
    return df


df=remove_null_rows(df, TARGET_COL)


# In[ ]:


df['NewPrice'].isna().sum()


# ## Check Columns with Mixed D_Types

# In[ ]:


check_cols=[27, 91]
for c in check_cols:
    check_col=df.columns[c]
    print(check_col)
    print('D_Types are: ',set(type(s) for s in df[check_col]))


# In[ ]:


print('INFO - EngineDescription is just a rounded up number for EngineSize')
df[['EngineDescription','EngineSize']]
## ***Add Round Up Verification


# In[ ]:


print('INFO - Drop EngineDescription')
df=df.drop('EngineDescription', axis=1)
nrows, ncols = df.shape
print(f'INFO - Dataset contains {nrows} rows, {ncols} columns')


# In[ ]:


nulls_pct = df['NormalChargeVoltage'].isna().sum()/nrows*100
print(f'INFO - {round(nulls_pct,3)}% Nulls in NormalChargeVoltage')
print('INFO - Drop NormalChargeVoltage')

df=df.drop('NormalChargeVoltage', axis=1)
nrows, ncols = df.shape
print(f'INFO - Dataset contains {nrows} rows, {ncols} columns')


# ## Drop Columns with Majority Nulls

# In[ ]:


print('INFO - Columns with Nulls')
null_cols=df.isnull().sum(axis=0)
null_cols[null_cols>0]


# In[ ]:


print(f'INFO - Dropping Numerical Columns with Nulls Rows Larger than Cutoff {CUTOFF_NULL_PCT}%')
drop_cols = null_cols[null_cols> nrows * CUTOFF_NULL_PCT / 100]
print(f'INFO - Dropping {len(drop_cols)}/{len(df.columns)} Numerical Columns due to Nulls')
print(drop_cols)


# In[ ]:


df


# # End of Preprocess 1

# In[ ]:


df.to_csv(OUTPUT_DATASET_PATH, index=None)
print(f'INFO - End of Preprocess 1. Saved dataset to {OUTPUT_DATASET_PATH}')


# In[13]:


# pd.read_csv('../datasets/dataset_1.csv')


# In[ ]:




