#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import Counter

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier,VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,learning_curve

sns.set(style = 'white',context = 'notebook',palette = 'deep')


# In[3]:


train = pd.read_csv("../data/titanic/train.csv")
test = pd.read_csv("../data/titanic/test.csv")
IDtest = test["PassengerId"]


# In[6]:


def detect_outliers(df,n,featuers):
    
    outlier_indices = []
    for col in featuers :
        
        Q1 = np.percentile(df[col],25)
        Q3 = np.percentile(df[col],75)
        IQR = Q3 - Q1
        
        outlier_step = 1.5 * IQR
        
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        
        outlier_indices.extend(outlier_list_col)
        
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k,v in outlier_indices.items() if v > n)
    
    return multiple_outliers

Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])
    


# In[7]:


train.loc[Outliers_to_drop]


# In[8]:


train = train.drop(Outliers_to_drop,axis = 0).reset_index(drop = True)


# In[9]:


train_len = len(train)
dataset = pd.concat(objs = [train,test],axis = 0).reset_index(drop = True)


# In[10]:


dataset = dataset.fillna(np.nan)
dataset.isnull().sum()


# In[11]:


train.info()
train.isnull().sum()


# In[12]:


train.head()


# In[14]:


train.dtypes


# In[15]:


train.describe()


# In[ ]:




