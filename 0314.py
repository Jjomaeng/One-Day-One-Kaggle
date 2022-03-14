#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from numba import jit
import time
import gc


# In[2]:


train_df = pd.read_csv('../data/porto-seguro-safe-driver-prediction/train.csv',na_values = '-1')
test_df = pd.read_csv('../data/porto-seguro-safe-driver-prediction/test.csv',na_values = '-1')


# In[ ]:




