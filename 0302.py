#!/usr/bin/env python
# coding: utf-8

# In[112]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns',100)


# In[113]:


train = pd.read_csv('../data/porto-seguro-safe-driver-prediction/train.csv')
test = pd.read_csv('../data/porto-seguro-safe-driver-prediction/test.csv')


# In[114]:


train.head()


# In[115]:


train.shape


# In[116]:


train.drop_duplicates() # feature가 많기 때문에 중복되는 feature가 많은지 확인
train.shape


# In[117]:


test.shape


# In[118]:


train.info()


# In[119]:


data = []
for f in train.columns:
    if f == "target":
        role = "target"
    elif f == "id":
        role = 'id'
    else :
        role = 'input'
        
    if 'bin' in f or f == "target":
        level = "binary"
    elif 'cat' in f or f == 'id':
        level = 'nominal'
    elif train[f].dtype == float:
        level = 'interval'
    elif train[f].dtype == int:
        level = 'ordinal'
        
    keep = True
    if f == 'id':
        keep = False
        
    dtype = train[f].dtype
    
    f_dict = {
        
        'varname' : f,
        'role':role,
        'level':level,
        'keep':keep,
        'dtype':dtype
    }
    data.append(f_dict)
    
meta = pd.DataFrame(data,columns = ['varname','role','level','keep','dtype'])
meta.set_index('varname',inplace = True)


# In[120]:


meta


# In[121]:


meta[(meta.level == 'nominal') & (meta.keep)].index


# In[122]:


pd.DataFrame({'count':meta.groupby(['role','level']).size()}).reset_index()


# In[123]:


v = meta[(meta.level == 'interval') & (meta.keep)].index
train[v].describe()


# In[124]:


v = meta[(meta.level == 'ordinal') & meta.keep].index
train[v].describe()


# In[125]:


v = meta[(meta.level == 'binary') & meta.keep].index
train[v].describe()


# In[126]:


desired_apriori = 0.1

idx_0 = train[train['target'] == 0].index
idx_1 = train[train['target'] == 1].index

nb_0 = len(train.loc[idx_0])
nb_1 = len(train.loc[idx_1])

undersampling_rate = ((1-desired_apriori) * nb_1) / nb_0 * desired_apriori
undersampled_nb_0 = int(undersampling_rate * nb_0)

undersampled_idx = shuffle(idx_0,random_state=37,n_samples=undersampled_nb_0)

idx_list = list(undersampled_idx) + list(idx_1)

train = train.loc[idx_list].reset_index(drop = True)


# In[127]:


vars_with_missing = []

for f in train.columns: 
    missings = train[train[f] == -1][f].count() # [f]를 붙이지 않으면 모든 column을 count해서 series로 반환
    if missings > 0 :
        vars_with_missing.append(f)
        missing_perc = missings/train.shape[0]
        print("Variable {} has {} records ({:.2%}) with missing values".format(f,missings,missing_perc))
        
print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))


# In[128]:


vars_to_drop = ['ps_car_03_cat','ps_car_05_cat']
train.drop(vars_to_drop,inplace = True,axis = 1)
meta.loc[vars_to_drop,'keep'] = False

mean_imp = SimpleImputer(missing_values=-1,strategy='mean')
mode_imp = SimpleImputer(missing_values=-1,strategy='most_frequent')

train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel() # ravel() 안 하면 np.array로 반환
train['ps_car_12'] = mean_imp.fit_transform(train[['ps_car_12']]).ravel()
train['ps_car_14'] = mean_imp.fit_transform(train[['ps_car_14']]).ravel()
train['ps_car_11'] = mode_imp.fit_transform(train[['ps_car_11']]).ravel()


# In[129]:


v = meta[(meta.level == 'nominal') & meta.keep].index

for f in v:
    dist_values = train[f].value_counts().shape[0]
    print(f"Variable {f} has {dist_values} distinct values" )


# In[134]:


def add_noise(series,noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series = None,
                  tst_series = None,
                  target = None,
                  min_samples_leaf = 1,
                  smoothing = 1,
                  noise_level = 0):
    
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series,target],axis = 1)
    
    averages = temp.groupby(by = trn_series.name)[target.name].agg(['mean','count']) # trn category 별로 평균값과 개수
    
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    
    prior = target.mean()
    
    
    
    averages[target.name] = prior * ( 1- smoothing) + averages['mean']*smoothing
    
    averages.drop(['mean','count'],axis = 1, inplace = True)
    

    ft_trn_series = pd.merge(trn_series.to_frame(trn_series.name),
                             averages.reset_index().rename(columns = {'index': target.name,target.name:'average'}),
                            on = trn_series.name,
                            how = 'left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(tst_series.to_frame(trn_series.name),
                             averages.reset_index().rename(columns = {'index':target.name,target.name: 'average'}),
                            on = tst_series.name,
                            how = 'left')['average'].rename(tst_series.name + '_mean').fillna(prior)
    
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series,noise_level)


# In[140]:


train_encoded,test_encoded = target_encode(train['ps_car_11_cat'],test['ps_car_11_cat'],
                                          target = train.target,
                                          min_samples_leaf = 100,
                                          smoothing = 10,
                                          noise_level = 0.01)

train['ps_car_11_cat_te'] = train_encoded
train.drop('ps_car_11_cat',axis = 1, inplace = True)
meta.loc['ps_car_11_cat','keep'] = False
test['ps_car_11_cat_te'] = test_encoded
test.drop('ps_car_11_cat',axis = 1,inplace = True)


# In[142]:


v = meta[(meta.level == 'nominal') & meta.keep].index

for f in v :
    plt.figure()
    fig,ax = plt.subplots(figsize = (20,10))
    cat_perc = train[[f,'target']].groupby([f],as_index= False).mean()
    cat_perc.sort_values(by= 'target',ascending = False,inplace = True)
    
    sns.barplot(ax = ax, x = f, y = 'target',data = cat_perc,order = cat_perc[f])
    plt.ylabel('% target',fontsize = 18)
    plt.xlabel(f,fontsize = 18)
    plt.tick_params(axis = 'both',which = 'major',labelsize = 18)
    plt.show()


# In[144]:


def corr_heatmap(v):
    correlations = train[v].corr()
    
    cmap = sns.diverging_palette(220,10,as_cmap = True)
    
    fig,ax = plt.subplots(figsize = (10,10))
    sns.heatmap(correlations,cmap = cmap, vmax = 1.0,center = 0, 
                fmt = '.2f',square = True,linewidths = .5,annot = True,cbar_kws= {"shrink": .75} )
    plt.show()
    
v = meta[(meta.level == 'interval') & (meta.keep)].index
corr_heatmap(v)


# In[145]:


s = train.sample(frac = 0.1)


# In[146]:


sns.lmplot(x = 'ps_reg_02', y = 'ps_reg_03', data = s, hue = 'target',palette = "Set1",scatter_kws = {'alpha':0.3})
plt.show()


# In[147]:


sns.lmplot(x = 'ps_car_12', y = 'ps_car_13', data = s, hue = 'target',palette = "Set1",scatter_kws = {'alpha':0.3})
plt.show()


# In[148]:


sns.lmplot(x = 'ps_car_12', y = 'ps_car_14', data = s, hue = 'target',palette = "Set1",scatter_kws = {'alpha':0.3})
plt.show()


# In[150]:


sns.lmplot(x = 'ps_car_15', y = 'ps_car_13', data = s, hue = 'target',palette = "Set1",scatter_kws = {'alpha':0.3})
plt.show()


# In[151]:


v = meta[(meta.level == 'ordinal') & (meta.keep)].index
corr_heatmap(v)


# In[152]:


v = meta[(meta.level == 'nominal') & meta.keep].index
train = pd.get_dummies(train,columns = v, drop_first = True)


# In[153]:


v = meta[(meta.level == 'interval') & meta.keep].index
poly = PolynomialFeatures(degree = 2, interaction_only=False, include_bias=False)
interactions = pd.DataFrame(data = poly.fit_transform(train[v]),columns = poly.get_feature_names(v))
interactions.drop(v,axis = 1,inplace = True)


# In[154]:


train = pd.concat([train,interactions],axis = 1)


# In[155]:


selector = VarianceThreshold(threshold=.01)
selector.fit(train.drop(['id','target'],axis = 1))

f = np.vectorize(lambda x : not x)

v = train.drop(['id','target'],axis = 1).columns[f(selector.get_support())]


# In[156]:


X_train = train.drop(['id','target'],axis = 1)
y_train = train['target']

feat_labels = X_train.columns

rf = RandomForestClassifier(n_estimators=1000,random_state=0,n_jobs = -1)

rf.fit(X_train,y_train)
importances = rf.feature_importances_

indices = np.argsort(rf.feature_importances_)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f"%(f+1,30,feat_labels[indices[f]],importances[indices[f]]))# 전체 print할 문자열 수 30


# In[157]:


# get_support 메서드는 선택된 특성을 불리언 값으로 표시해주어 어떤 특성이 선택되었는지 확인 할 수 있다.
sfm = SelectFromModel(rf,threshold = 'median',prefit = True) # prefit 이미 fit을 했냐 여부
n_features = sfm.transform(X_train).shape[1]
selected_vars = list(feat_labels[sfm.get_support()])


# In[158]:


train = train[selected_vars + ['target']]


# In[159]:


scaler = StandardScaler()
scaler.fit_transform(train.drop(['target'],axis = 1))


# In[ ]:




