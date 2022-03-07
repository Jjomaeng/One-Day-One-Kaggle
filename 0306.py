#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


train = pd.read_csv('../data/porto-seguro-safe-driver-prediction/train.csv')
test = pd.read_csv('../data/porto-seguro-safe-driver-prediction/test.csv')


# In[4]:


train.isnull().any().any()


# In[5]:


train_copy = train
train_copy = train_copy.replace(-1,np.NaN)


# In[7]:


import missingno as msno
msno.matrix(df= train_copy.iloc[:,2:39],figsize = (20,14),color = (0.42,0.1,0.05))


# In[11]:


data = [go.Bar(
            x = train["target"].value_counts().index.values,
            y = train["target"].value_counts().values,
            text = "Distribution of target variable")]

layout = go.Layout(title = 'Target variable distribution')

fig = go.Figure(data = data,layout = layout)
py.iplot(fig,filename = 'basic-bar')


# In[12]:


Counter(train.dtypes.values)


# In[13]:


train.info()


# In[15]:


train_float = train.select_dtypes(include = ['float64'])
train_int = train.select_dtypes(include = ['int64'])


# In[16]:


colormap = plt.cm.magma
plt.figure(figsize=(16,12))
plt.title('Pearson correlation of continuous features', y=1.05, size=15)
sns.heatmap(train_float.corr(),linewidths=0.1,vmax=1.0, square=True, 
            cmap=colormap, linecolor='white', annot=True)


# In[19]:


data = [
    go.Heatmap(
        z= train_int.corr().values,
        x=train_int.columns.values,
        y=train_int.columns.values,
        colorscale='Viridis',
        reversescale = False,
        opacity = 1.0 )
]

layout = go.Layout(
    title='Pearson Correlation of Integer-type features',
    xaxis = dict(ticks='', nticks=36),
    yaxis = dict(ticks='' ),
    width = 900, height = 700)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='labelled-heatmap')


# In[20]:


mf = mutual_info_classif(train_float.values,train.target.values,n_neighbors=3,random_state=2022)
print(mf)


# In[21]:


bin_col = [col for col in train.columns if 'bin' in col]
zero_list = []
one_list = []
for col in bin_col :
    zero_list.append((train[col] == 0).sum())
    one_list.append((train[col] == 1).sum())


# In[23]:


trace1 = go.Bar(
    x = bin_col,
    y = zero_list,
    name = "Zero count")

trace2 = go.Bar(
    x = bin_col,
    y = one_list,
    name = "One count"
)

data = [trace1,trace2]

layout = go.Layout(
    barmode = 'stack',
    title = 'Count of 1 and 0 in binary variables')

fig = go.Figure(data = data,layout = layout)
py.iplot(fig,filename = 'stacked-bar')


# In[24]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=150,max_depth=8,min_samples_leaf=4,max_features=0.2,n_jobs = -1,random_state=0)
rf.fit(train.drop(['id','target'],axis = 1),train.target)
features = train.drop(['id','target'],axis = 1).columns.values


# In[25]:


trace = go.Scatter(
    y = rf.feature_importances_,
    x = features,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 13,
        #size= rf.feature_importances_,
        #color = np.random.randn(500), #set color equal to a variable
        color = rf.feature_importances_,
        colorscale='Portland',
        showscale=True
    ),
    text = features
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Random Forest Feature Importance',
    hovermode= 'closest',
     xaxis= dict(
         ticklen= 5,
         showgrid=False,
        zeroline=False,
        showline=False
     ),
    yaxis=dict(
        title= 'Feature Importance',
        showgrid=False,
        zeroline=False,
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')


# In[26]:


x, y = (list(x) for x in zip(*sorted(zip(rf.feature_importances_, features), 
                                                            reverse = False)))
trace2 = go.Bar(
    x=x ,
    y=y,
    marker=dict(
        color=x,
        colorscale = 'Viridis',
        reversescale = True
    ),
    name='Random Forest Feature importance',
    orientation='h',
)

layout = dict(
    title='Barplot of Feature importances',
     width = 900, height = 2000,
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
#         domain=[0, 0.85],
    ))

fig1 = go.Figure(data=[trace2])
fig1['layout'].update(layout)
py.iplot(fig1, filename='plots')


# In[27]:


from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, min_samples_leaf=4, max_features=0.2, random_state=0)
gb.fit(train.drop(['id', 'target'],axis=1), train.target)
features = train.drop(['id', 'target'],axis=1).columns.values


# In[28]:


trace = go.Scatter(
    y = gb.feature_importances_,
    x = features,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 13,
        #size= rf.feature_importances_,
        #color = np.random.randn(500), #set color equal to a variable
        color = gb.feature_importances_,
        colorscale='Portland',
        showscale=True
    ),
    text = features
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Gradient Boosting Machine Feature Importance',
    hovermode= 'closest',
     xaxis= dict(
         ticklen= 5,
         showgrid=False,
        zeroline=False,
        showline=False
     ),
    yaxis=dict(
        title= 'Feature Importance',
        showgrid=False,
        zeroline=False,
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')


# In[29]:


x, y = (list(x) for x in zip(*sorted(zip(gb.feature_importances_, features), 
                                                            reverse = False)))

trace2 = go.Bar(
    x=x ,
    y=y,
    marker=dict(
        color=x,
        colorscale = 'Viridis',
        reversescale = True
    ),
    name='Gradient Boosting Classifer Feature importance',
    orientation='h',
)

layout = dict(
    title='Barplot of Feature importances',
     width = 900, height = 2000,
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
    ))

fig1 = go.Figure(data=[trace2])
fig1['layout'].update(layout)
py.iplot(fig1, filename='plots')


# In[ ]:




