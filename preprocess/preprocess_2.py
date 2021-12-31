#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
from datetime import datetime


# In[78]:


pd.set_option("display.max_columns", None)


# In[79]:


def remove_null_rows(df, col):
    target_nulls = df[col].isna()
    if target_nulls.sum():
        print(f'WARNING - Found {target_nulls.sum()} Null Values in : {col}. Will drop rows')
        df = df[~target_nulls]
        nrows, ncols = df.shape
        print(f'INFO - Dataset contains {nrows} rows, {ncols} columns')   
    return df


# In[80]:


df=pd.read_csv('../datasets/dataset_1.csv')
print('INFO - Load Dataset from Preprocess 1')
df.head()


# # Missingness & Imputations

# In[81]:


SELECT_CAT_COLS = [
    'MakeCode', 'FamilyCode', 'BodyStyleDescription', 'DriveCode', 'GearTypeDescription',
    'GearLocationDescription', 'FuelTypeDescription', 'InductionDescription', 'BuildCountryOriginDescription', 
]
SELECT_NUM_COLS = [
    'GearNum', 'DoorNum', 'EngineSize', 'Cylinders', 'FuelCapacity', 'NewPrice',
    'WarrantyYears', 'WarrantyKM', 'KM',
]
SELECT_DATE_COLS = ['Sold_Date', 'YearGroup','MonthGroup', 'Compliance_Date',  'Age_Comp_Months']


# In[82]:


df[SELECT_CAT_COLS].isna().sum()


# In[83]:


col='GearLocationDescription'
df[col]=df[col].fillna(df[col].mode().values[0])


# In[84]:


df[SELECT_CAT_COLS].isna().sum()


# In[85]:


df[SELECT_NUM_COLS].isna().sum()


# In[86]:


for col in ['GearNum', 'WarrantyYears', 'WarrantyKM']:
    df[col]=df[col].fillna(df[col].mode().values[0])
    
df['FuelCapacity'].fillna(df['FuelCapacity'].mean(), inplace=True)
    
df=remove_null_rows(df, 'KM')
df=remove_null_rows(df, 'NewPrice')


# In[87]:


df[SELECT_NUM_COLS].isna().sum()


# In[88]:


df[SELECT_DATE_COLS].isna().sum()


# In[89]:


df=remove_null_rows(df, 'Compliance_Date')


# In[90]:


df[SELECT_DATE_COLS].isna().sum()


# # Date / Duration

# In[91]:


print('INFO : Age Comp Months Stats')
print(df['Age_Comp_Months'].describe())


# In[92]:


print('INFO : Check Age_Comp_Months = 0')
print(df[df['Age_Comp_Months']==0][['Compliance_Date','Sold_Date','Age_Comp_Months']])
print('INFO : Seems like sold within 1 Month. Will use Age Days for better resolution')


# In[93]:


df['Sold_Date']=pd.to_datetime(df['Sold_Date'])
df['Compliance_Date']=pd.to_datetime(df['Compliance_Date'])


# In[94]:


df['AgeDays']=df['Sold_Date']-df['Compliance_Date']
print('INFO - Derived AgeDays feature')
df['AgeDays']
# df['YearGroup'].value_counts()


# In[95]:


print('INFO : Check Age_Comp_Months > 1000')
print(df[df['Age_Comp_Months']>1000][['Compliance_Date','Sold_Date','Age_Comp_Months']])


# In[96]:


bad_comp_date = df['Compliance_Date']=='1900-01-01'
print(f'INFO : Remove {sum(bad_comp_date)} Rows with Compliance_Date==1900-01-01')
df=df[~bad_comp_date]


# # Feature Engineering - ValidWarranty

# In[97]:


df['WarrantyDays']=pd.to_timedelta(df['WarrantyYears']*365, unit='D')


# In[98]:


df['ValidWarranty'] = ( df['AgeDays'] < df['WarrantyDays'] ) & ( df['KM'] < df['WarrantyKM'] )


# In[99]:


print(f"INFO - {df['ValidWarranty'].sum()} / {df.shape[0]} Vehicles Under Warranty (Feature Eng)")


# In[100]:


#Change AgeDays from Timedelta to Int
df['AgeDays']=df['AgeDays'].dt.days


# # Category Remap

# In[101]:


df['FuelTypeDescription'].value_counts()


# In[102]:


gear_type_map={
    'Automatic':'Automatic',
    'Sports Automatic': 'Sports Automatic',
    'Manual' : 'Manual',
    'Constantly Variable Transmission' : 'Constantly Variable Transmission',
    'Sports Automatic Dual Clutch' : 'Sports Automatic',
    'Sports Automatic Single Clutch' : 'Sports Automatic',
    'Seq. Manual Auto-Single Clutch' : 'Manual',
    'Manual Auto-clutch - H Pattern' : 'Manual',
}

induction_map={
    'Aspirated':'Aspirated',
    'Turbo Intercooled' :'Turbo',                 
    'Turbo' :'Turbo',                              
    'Twin Turbo Intercooled' : 'Turbo',              
    'Supercharged' : 'Supercharged',                       
    'Supercharged Intercooled' : 'Supercharged',            
    'Turbo Supercharged Intercooled' : 'Turbo Supercharged',     
}
fuel_type_map={
    'Petrol - Unleaded ULP' :'Petrol',
    'Diesel' : 'Diesel',
    'Petrol - Premium ULP' :'Petrol',
    'LPG only' : 'LPG',
    'Petrol or LPG (Dual)' : 'Petrol or LPG',
    'Petrol': 'Petrol',
}


# In[103]:


df['GearTypeDescription'] = df['GearTypeDescription'].apply(lambda x: gear_type_map[x])
df['InductionDescription'] = df['InductionDescription'].apply(lambda x: induction_map[x])
df['FuelTypeDescription'] = df['FuelTypeDescription'].apply(lambda x: fuel_type_map[x])


# In[104]:


SELECT_CAT_COLS = [
    'MakeCode', 'FamilyCode', 'BodyStyleDescription', 'DriveCode', 'GearTypeDescription',
    'GearLocationDescription', 'FuelTypeDescription', 'InductionDescription', 'ValidWarranty',
    'BuildCountryOriginDescription'
]
SELECT_NUM_COLS = [
    'GearNum', 'DoorNum', 'EngineSize', 'Cylinders', 'FuelCapacity', 'NewPrice','KM',
]
SELECT_TIMEDELTA_COLS = ['AgeDays']
SELECT_TARGET_COL = ['Sold_Amount']


# In[105]:


df=df[SELECT_CAT_COLS + SELECT_NUM_COLS + SELECT_TIMEDELTA_COLS + SELECT_TARGET_COL]
df


# In[106]:


df.isna().sum()


# In[107]:


df.to_csv('../datasets/dataset_2.csv', index=None)


# # Categorical Label Encoding

# In[108]:


df=pd.read_csv('../datasets/dataset_2.csv')


# In[110]:


for col in SELECT_CAT_COLS:
    print(col)
    df_ = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,df_], axis=1)
    df.drop(col, axis=1, inplace=True)


# # Apply np.log(x+1) for Sold_Amount & NewPrice

# In[109]:


df['Sold_Amount'] = df['Sold_Amount'].apply(lambda x: np.log(x+1))
df['NewPrice'] = df['NewPrice'].apply(lambda x: np.log(x+1))


# In[111]:


df.shape


# # Train Test Split

# In[114]:


from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.15, random_state=42)
print(f'INFO - Size Train {len(df_train)}')
print(f'INFO - Size Test {len(df_test)}')


# In[113]:


print('INFO - Save Train & Test sets')
df_train.to_csv('../datasets/df_train.csv', index=None)
df_test.to_csv('../datasets/df_test.csv', index=None)

