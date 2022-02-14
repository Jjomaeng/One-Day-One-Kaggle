#!/usr/bin/env python
# coding: utf-8

# In[81]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns',100)


# In[82]:


train = pd.read_csv('../data/porto-seguro-safe-driver-prediction/train.csv')
test = pd.read_csv('../data/porto-seguro-safe-driver-prediction/test.csv')


# In[83]:


train.head(10)


# In[84]:


train.info()


# In[85]:


train.describe()


# ## target

# In[86]:


f,ax = plt.subplots(1,2,figsize = (18,15))

train['target'].value_counts().plot.pie(autopct = "%1.1f%%",ax = ax[0])
sns.countplot('target',data = train)


# In[87]:


data = []
for f in train.columns:
    
    if f == 'target':
        role = 'target'
    elif f == 'id':
        role = 'id'
    else :
        role = 'input'
        
    
    if 'bin' in f or f == 'target':
        level = 'binary'
    elif 'cat' in f or f == 'id':
        level = 'nominal'
    elif train[f].dtype == float :
        level = 'interval'
    elif train[f].dtype == int :
        level = 'ordinal'
    
    keep = True
    if f == 'id':
        keep = False
    
    dtype = train[f].dtype
    
    f_dict = {
        'varname' : f,
        'role' : role,
        'level': level,
        'keep' : keep,
        'dtype' : dtype
        
    }
    data.append(f_dict)
    
    
meta = pd.DataFrame(data,columns = ['varname','role','level','keep','dtype'])
meta.set_index('varname',inplace = True)


# In[88]:


meta


# In[89]:


meta[(meta.level == 'nominal') & (meta.keep)].index


# In[90]:


pd.DataFrame({'count':meta.groupby(['role','level'])['role'].size()}).reset_index()


# In[91]:


v = meta[(meta.level == 'interval') & (meta.keep)].index
train[v].describe()


# In[92]:


v = meta[(meta.level == 'ordinal') & (meta.keep)].index
train[v].describe()


# In[93]:


v = meta[(meta.level == 'binary') & (meta.keep)].index
train[v].describe()


# In[94]:


desired_apriori = 0.10

idx_0 = train[train.target == 0].index
idx_1 = train[train.target == 1].index

nb_0 = len(train.loc[idx_0])
nb_1 = len(train.loc[idx_1])

undersampling_rate = ((1-desired_apriori)*nb_1)/ (nb_0*desired_apriori)
undersampled_nb_0 = int(undersampling_rate*nb_0)
print('Rate to undersample records with target=0: {}'.format(undersampling_rate))
print('Number of records with target=0 before : {}, after undersampling: {}'.format(nb_0,undersampled_nb_0))

undersampled_idx = shuffle(idx_0,random_state = 37,n_samples = undersampled_nb_0)

idx_list = list(undersampled_idx) + list(idx_1)

train = train.loc[idx_list].reset_index(drop=True)


# In[95]:


vars_with_missing = []

for f in train.columns:
    missings = train[train[f] == -1][f].count()
    if missings > 0:
        vars_with_missing.append(f)
        missings_perc = missings/train.shape[0]
        
        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))
        
print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))


# In[96]:


vars_to_drop = ['ps_car_03_cat','ps_car_05_cat']
train.drop(vars_to_drop,inplace = True, axis = 1)
meta.loc[(vars_to_drop),'keep'] = False

mean_imp = SimpleImputer(missing_values = -1,strategy = 'mean')
mode_imp = SimpleImputer(missing_values = -1,strategy = 'most_frequent')
train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()
train['ps_car_12'] = mean_imp.fit_transform(train[['ps_car_12']]).ravel()
train['ps_car_14'] = mode_imp.fit_transform(train[['ps_car_14']]).ravel()
train['ps_car_11'] = mode_imp.fit_transform(train[['ps_car_11']]).ravel()


# In[100]:


v = meta[(meta.level == 'nominal') & (meta.keep)].index

for f in v:
    dist_values = train[f].value_counts().shape[0]
    print('Variable {} has {} distinct values'.format(f, dist_values))


# In[ ]:




