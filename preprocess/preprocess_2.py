#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[266]:


pd.set_option("display.max_columns", None)


# In[267]:


def remove_null_rows(df, col):
    target_nulls = df[col].isna()
    if target_nulls.sum():
        print(f'WARNING - Found {target_nulls.sum()} Null Values in : {col}. Will drop rows')
        df = df[~target_nulls]
        nrows, ncols = df.shape
        print(f'INFO - Dataset contains {nrows} rows, {ncols} columns')   
    return df


# In[268]:


df=pd.read_csv('../datasets/dataset_1.csv')
# print('INFO - Load Dataset Preprocess 1 from')
df.head()


# # Select Features

# In[269]:


SELECT_CAT_COLS = [
    'MakeCode', 'FamilyCode', 'BodyStyleDescription', 'DriveCode', 'GearTypeDescription',
    'GearLocationDescription', 'FuelTypeDescription', 'InductionDescription',  
]
SELECT_NUM_COLS = [
    'GearNum', 'DoorNum', 'EngineSize', 'Cylinders', 'FuelCapacity', 'NewPrice',
    'WarrantyYears', 'WarrantyKM', 'KM'
]
SELECT_DATE_COLS = ['Sold_Date', 'YearGroup','MonthGroup', 'Compliance_Date']


# In[270]:


df[SELECT_CAT_COLS].isna().sum()


# In[271]:


col='GearLocationDescription'
df[col]=df[col].fillna(df[col].mode().values[0])


# In[272]:


df[SELECT_CAT_COLS].isna().sum()


# In[273]:


df[SELECT_NUM_COLS].isna().sum()


# In[274]:


for col in ['GearNum', 'WarrantyYears', 'WarrantyKM']:
    df[col]=df[col].fillna(df[col].mode().values[0])
    
df['FuelCapacity'].fillna(df['FuelCapacity'].mean(), inplace=True)
    
df=remove_null_rows(df, 'KM')
df=remove_null_rows(df, 'NewPrice')


# In[275]:


df[SELECT_NUM_COLS].isna().sum()


# In[276]:


df[SELECT_DATE_COLS].isna().sum()


# In[277]:


df=remove_null_rows(df, 'Compliance_Date')


# In[278]:


df[SELECT_DATE_COLS].isna().sum()


# # Feature Eng

# In[279]:


# Age Build Months
from datetime import datetime


# In[280]:


df['Sold_Date']=pd.to_datetime(df['Sold_Date'])
df['Compliance_Date']=pd.to_datetime(df['Compliance_Date'])


# In[281]:


# df['Sold_Year']=df['Sold_Date'].apply(lambda x:x.year)


# In[282]:


df['AgeDays']=df['Sold_Date']-df['Compliance_Date']
df['AgeDays']
# df['YearGroup'].value_counts()


# In[283]:


df['WarrantyDays']=pd.to_timedelta(df['WarrantyYears']*365, unit='D')


# In[284]:


df['ValidWarranty'] = ( df['AgeDays'] < df['WarrantyDays'] ) & ( df['KM'] < df['WarrantyKM'] )


# In[285]:


print(f"INFO - {df['ValidWarranty'].sum()} / {df.shape[0]} Vehicles Under Warranty")


# # Remap

# In[286]:


df['FuelTypeDescription'].value_counts()


# In[287]:


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


# In[288]:


df['GearTypeDescription'] = df['GearTypeDescription'].apply(lambda x: gear_type_map[x])
df['InductionDescription'] = df['InductionDescription'].apply(lambda x: induction_map[x])
df['FuelTypeDescription'] = df['FuelTypeDescription'].apply(lambda x: fuel_type_map[x])


# In[289]:


SELECT_CAT_COLS = [
    'MakeCode', 'FamilyCode', 'BodyStyleDescription', 'DriveCode', 'GearTypeDescription',
    'GearLocationDescription', 'FuelTypeDescription', 'InductionDescription', 'ValidWarranty' 
]
SELECT_NUM_COLS = [
    'GearNum', 'DoorNum', 'EngineSize', 'Cylinders', 'FuelCapacity', 'NewPrice',
    'KM'
]
SELECT_TIMEDELTA_COLS = ['AgeDays']
SELECT_TARGET_COL = ['Sold_Amount']


# In[290]:


df=df[SELECT_CAT_COLS + SELECT_NUM_COLS + SELECT_TIMEDELTA_COLS + SELECT_TARGET_COL]
df


# In[291]:


df.isna().sum()


# In[292]:


for col in SELECT_CAT_COLS:
    print(col)
    df_ = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,df_], axis=1)
    df.drop(col, axis=1, inplace=True)


# In[293]:


#Change AgeDays from Timedelta to Int
df['AgeDays']=df['AgeDays'].dt.days


# In[294]:


df.shape


# In[296]:


df.to_csv('../datasets/dataset_2.csv', index=None)


# In[2]:


df=pd.read_csv('../datasets/dataset_2.csv')


# In[4]:


from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.15, random_state=42)
print(len(df_train))
print(len(df_test))


# In[6]:


df_train.to_csv('../datasets/df_train.csv', index=None)
df_test.to_csv('../datasets/df_test.csv', index=None)

