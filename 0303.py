#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected = True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
from collections import Counter
from sklearn.feature_selection import mutual_info_classif
warnings.filterwarnings('ignore')


# In[4]:


train = pd.read_csv('../data/porto-seguro-safe-driver-prediction/train.csv')
test = pd.read_csv('../data/porto-seguro-safe-driver-prediction/test.csv')


# In[5]:


train.head()


# In[6]:


rows = train.shape[0]
columns = train.shape[1]
train.shape


# In[8]:


#전달받은 자료형의 element 중 하나라도 True일 경우 True를 돌려준다. 
#(만약 empty 값을 argument로 넘겨주었다면 False를 돌려준다.)

train.isnull().any().any() # any()한 번이면 컬럼별로 두 번은 컬럼 전체


# In[9]:


train_copy = train
train_copy = train_copy.replace(-1,np.NaN)


# In[11]:


import missingno as msno
msno.matrix(df=train_copy.iloc[:,2:39],figsize = (20,14))


# In[12]:


data = [go.Bar( x= train['target'].value_counts().index.values,
              y = train['target'].value_counts().values,
              text = 'Distribution of target variable')]

layout = go.Layout(
    title = 'Target variable distribution')

fig = go.Figure(data = data,layout = layout)

py.iplot(fig,filename = 'basic -bar')


# In[13]:


Counter(train.dtypes.values)


# In[14]:


train_float = train.select_dtypes(include = ['float64'])
train_int = train.select_dtypes(include = ['int64'])


# In[15]:


colormap = plt.cm.magma
plt.figure(figsize = (16,12))
plt.title('Pearson correlation of continuous features',y = 1.05,size = 15)
sns.heatmap(train_float.corr(),linewidths = 0.1,vmax = 1.0,square = True,cmap = colormap,linecolor = 'white',annot =True
)


# In[18]:


data = [
    go.Heatmap(
    z = train_int.corr().values,
    x = train_int.columns.values,
    y = train_int.columns.values,
    colorscale = 'Viridis',
    reversescale = False,
    opacity=1.0)
]

layout = go.Layout(
    title= 'Pearson Correlation of Integer-type features',
    xaxis = dict(ticks = '', nticks = 36),
    yaxis= dict(ticks = ''),
    width = 900, height = 700)

fig = go.Figure(data = data,layout = layout)
py.iplot(fig,filename = 'labelled-heatmap')


# In[ ]:




