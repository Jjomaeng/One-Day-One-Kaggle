#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
py.init_notebook_mode(connected = True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import KFold


# In[2]:


train = pd.read_csv("../data/titanic/train.csv")
test = pd.read_csv("../data/titanic/test.csv")

PassengerId = test["PassengerId"]

train.head(3)


# In[3]:


full_data = [train,test]

train["Name_length"] = train['Name'].apply(len)
test["Name_length"] = test["Name"].apply(len)

train["Has_Cabin"] = train["Cabin"].apply(lambda x :0 if type(x) == float else 1 )
test["Has_Cabin"] = test["Cabin"].apply(lambda x :0 if type(x) == float else 1 )

for dataset in full_data:
    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1

for dataset in full_data:
    dataset["IsAlone"] = 0
    dataset.loc[dataset["FamilySize"] == 1,'IsAlone'] =1
    
for dataset in full_data:
    dataset["Embarked"] = dataset["Embarked"].fillna("S")

for dataset in full_data:
    dataset["Fare"] = dataset["Fare"].fillna(train['Fare'].median())
train["CategoricalFare"] = pd.qcut(train["Fare"],4)


for dataset in full_data:
    age_avg = dataset["Age"].mean()
    age_std = dataset["Age"].std()
    age_null_count = dataset["Age"].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std,age_avg+age_std, size = age_null_count)
    dataset["Age"][np.isnan(dataset["Age"])] = age_null_random_list
    dataset["Age"] = dataset["Age"].astype(int)
train['CategoricalAge'] = pd.cut(train["Age"],5)

def get_title(name):
    title_search = re.search('([A-Za-z]+)\.',name)
    
    if title_search :
        return title_search.group(1) # 첫번째 그룹에 해당하는 문자열 반환
    return ""

for dataset in full_data:
    dataset['Title'] = dataset["Name"].apply(get_title)
    
for dataset in full_data:
    dataset["Title"] = dataset["Title"].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
    
    dataset["Title"] = dataset["Title"].replace("Mlle","Miss")
    dataset["Title"] = dataset["Title"].replace("Ms","Miss")
    dataset["Title"] = dataset["Title"].replace("Mme","Mrs")

for dataset in full_data:
    
    dataset['Sex'] = dataset['Sex'].map({'female':0,'male':1}).astype(int)
    
    title_mapping = {'Mr':1,'Miss':2,"Mrs":3,"Master":4,"Rare":5}
    dataset['Title'] = dataset["Title"].map(title_mapping)
    dataset["Title"] = dataset["Title"].fillna(0)

    dataset["Embarked"] = dataset["Embarked"].map({'S':0,"C":1,"Q":2}).astype(int)
    
    dataset.loc[dataset["Fare"] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset["Fare"] <= 14.454),'Fare'] =1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset["Fare"] <= 31),'Fare'] =2
    dataset.loc[dataset["Fare"] > 31, 'Fare'] = 3
    dataset["Fare"] = dataset['Fare'].astype(int)
    
    dataset.loc[dataset['Age'] <=16,'Age'] = 0
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 32),'Age'] = 1
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64),'Age'] = 2
    dataset.loc[dataset['Age'] <= 64,'Age'] = 3


# In[4]:


drop_elements = ["PassengerId",'Name','Ticket','Cabin','SibSp']
train = train.drop(drop_elements,axis = 1)
train = train.drop(['CategoricalAge','CategoricalFare'],axis = 1)
test = test.drop(drop_elements, axis = 1)


# In[5]:


train.head()


# In[6]:


colormap = plt.cm.RdBu
plt.figure(figsize = (14,12))
plt.title('Personn Correlation of Feature', y = 1.05,size = 15)
sns.heatmap(train.astype(float).corr(),linewidths = 0.1,vmax = 1.0,square = True,cmap = colormap,linecolor = 'white',annot = True)


# In[7]:


g = sns.pairplot(train[[u'Survived',u'Pclass',u'Sex',u'Age',u'Parch',u'Fare',u'Embarked',u'FamilySize',u'Title']],hue = 'Survived',palette = 'seismic',size = 1.2,diag_kind='kde',diag_kws = dict(shade = True),plot_kws=dict(s = 10))
g.set(xticklabels = [])


# # modeling

# In[27]:


ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0
NFOLDS = 5
kf = KFold(n_splits = NFOLDS)

class SklearnHelper(object):
    def __init__(self,clf,seed = 0,params = None):
        params['random_state'] = seed
        self.clf = clf(**params)
    
    def train(self,x_train,y_train):
        self.clf.fit(x_train,y_train)
        
    def predict(self,x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return selfb.clf.fit(x,y)
    
    def feature_importance(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
    


# In[28]:


ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(n_splits= NFOLDS)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        
        print(self.clf.fit(x,y).feature_importances_)


# In[29]:


def get_oof(clf,x_train,y_train,x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS,ntest))
    
    for i ,(train_index,test_index) in enumerate(kf.split(train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        
        clf.train(x_tr,y_tr)
        
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i,:] = clf.predict(x_test)
        
    oof_test[:] = oof_test_skf.mean(axis = 0)
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)


# In[19]:


# warm start : When set to True, reuse the solution of the previous call to fit 
#              and add more estimators to the ensemble, 
#              otherwise, just fit a whole new forest. 

#max_features{“auto”, “sqrt”, “log2”}, int or float, default=”auto”
#The number of features to consider when looking for the best split:

#If int, then consider max_features features at each split.
#If float, then max_features is a fraction and round(max_features * n_features) features are considered at each split.
#If “auto”, then max_features=sqrt(n_features).
#If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
#If “log2”, then max_features=log2(n_features).
#If None, then max_features=n_features.

rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }


# In[20]:


rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)


# In[21]:


y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values 
x_test = test.values


# In[30]:


et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) 
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test)
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) 

print("Training is complete")


# In[31]:


rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)


# In[32]:


rf_features = [0.10474135,  0.21837029,  0.04432652,  0.02249159,  0.05432591,  0.02854371
  ,0.07570305,  0.01088129 , 0.24247496,  0.13685733 , 0.06128402]
et_features = [ 0.12165657,  0.37098307  ,0.03129623 , 0.01591611 , 0.05525811 , 0.028157
  ,0.04589793 , 0.02030357 , 0.17289562 , 0.04853517,  0.08910063]
ada_features = [0.028 ,   0.008  ,      0.012   ,     0.05866667,   0.032 ,       0.008
  ,0.04666667 ,  0.     ,      0.05733333,   0.73866667,   0.01066667]
gb_features = [ 0.06796144 , 0.03889349 , 0.07237845 , 0.02628645 , 0.11194395,  0.04778854
  ,0.05965792 , 0.02774745,  0.07462718,  0.4593142 ,  0.01340093]


# In[34]:


cols = train.columns.values

feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_features,
     'Extra Trees  feature importances': et_features,
      'AdaBoost feature importances': ada_features,
    'Gradient Boost feature importances': gb_features
    })


# In[36]:


# sizeref : To scale the bubble size
trace = go.Scatter( 
    y = feature_dataframe["Random Forest feature importances"].values, 
    x = feature_dataframe['features'].values,
    mode = 'markers',
    marker = dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['Random Forest feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Random Forest Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

trace = go.Scatter(
    y = feature_dataframe['Extra Trees  feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['Extra Trees  feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Extra Trees Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

trace = go.Scatter(
    y = feature_dataframe['AdaBoost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['AdaBoost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'AdaBoost Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

trace = go.Scatter(
    y = feature_dataframe['Gradient Boost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['Gradient Boost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]


layout= go.Layout(
    autosize= True,
    title= 'Gradient Boosting Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')


# In[37]:


feature_dataframe['mean'] = feature_dataframe.mean(axis = 1)
feature_dataframe.head(3)


# In[41]:


y = feature_dataframe['mean'].values
x = feature_dataframe['features'].values
data = [go.Bar(
            x = x,
            y = y,
            width = 0.5,
            marker = dict(
                color = feature_dataframe['mean'].values,
                colorscale = 'PortLand',
                showscale = True,
                reversescale = False
            ),
            opacity = 0.6
    )]
layout = go.Layout(
    autosize = True,
    title = 'Barplots of Mean Feature Importance',
    hovermode = 'closest',
    yaxis = dict(
        title = "Feature Importance",
        ticklen = 5,
        gridwidth = 2
    ),
    showlegend = False
)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig,filename = 'bar-direct- labels')


# In[ ]:




