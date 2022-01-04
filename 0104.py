#!/usr/bin/env python
# coding: utf-8

# In[132]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[133]:


df_train = pd.read_csv("../data/titanic/train.csv")
df_test = pd.read_csv("../data/titanic/test.csv")


# In[134]:


df_train.describe()


# # 데이터 확인( 전체 - NULL)

# In[135]:


df_train.info()


# In[136]:


for col in df_train.columns:
    msg = " {:>10}\t : {} % ".format(col,df_train[col].isnull().sum() / df_train[col].shape[0])
    print(msg)


# In[137]:


for col in df_test.columns:
    msg = "{:>10}\t : {}%".format(col,df_test[col].isnull().sum()/df_test[col].shape[0])
    print(msg)


# In[138]:


msno.matrix(df=df_train,figsize = (8,8))


# In[139]:


msno.bar(df = df_train,figsize = (8,8))


# In[140]:


msno.bar(df = df_test,figsize = (8,8))


# # Target Label 확인

# In[141]:


f,ax = plt.subplots(1,2,figsize = (18,8))
df_train["Survived"].value_counts().plot.pie(ax = ax[0],autopct = "%1.1f%%")
sns.countplot("Survived",data = df_train,ax = ax[1])


# # EDA

# ## - Pclass

# In[142]:


df_train[["Pclass","Survived"]].groupby(["Pclass"]).count()
df_train[["Pclass","Survived"]].groupby(["Pclass"]).sum()


# In[143]:


pd.crosstab(df_train["Pclass"],df_train["Survived"],margins = True)


# In[144]:


df_train[["Pclass","Survived"]].groupby(["Pclass"]).mean().sort_values(by = "Survived",ascending = False).plot.bar()


# In[145]:


f,ax = plt.subplots(1,2,figsize = (18,8))
sns.countplot("Pclass",data = df_train, ax = ax[0])
sns.countplot("Pclass",hue = "Survived",data = df_train,ax = ax[1])


# ## - Sex

# In[146]:


pd.crosstab(df_train["Sex"],df_train["Survived"],margins = True)


# In[147]:


f,ax = plt.subplots(1,2,figsize = (18,8))
df_train[["Sex","Survived"]].groupby(["Sex"]).mean().sort_values(by = "Survived",ascending = True).plot.bar(ax = ax[0])
sns.countplot("Sex",hue = "Survived",data = df_train,ax = ax[1])


# # -Both Sex and Pclass

# In[148]:


sns.factorplot("Pclass","Survived",hue = "Sex",data = df_train,size = 6, aspect = 1.5)


# In[149]:


sns.factorplot("Sex","Survived",col = "Pclass",data = df_train,satureation = .5,size = 9,aspect = 1)


# # - Age

# In[150]:


plt.figure(figsize = (9,5))
sns.kdeplot(df_train[df_train["Survived"] == 1]["Age"])
sns.kdeplot(df_train[df_train["Survived"] == 0]["Age"])
plt.legend(["Survived == 1","Survived == 0"])
plt.show()


# In[151]:


# pcalss별 age 분포
plt.figure(figsize = (9,5))
sns.kdeplot(df_train[df_train["Pclass"]==1]["Age"])
sns.kdeplot(df_train[df_train["Pclass"]==2]["Age"])
sns.kdeplot(df_train[df_train["Pclass"]==3]["Age"])
plt.legend(["Pclass1","Pclass2","Pclass3"])
plt.show()


# In[152]:


# cummulative age- survival rate
cummulative_age = []
for i in range(0,80):
    cummulative_age.append(df_train[df_train["Age"] < i]["Survived"].sum()/len(df_train[df_train["Age"]<i]["Survived"]))
    
plt.figure(figsize = (8,8))
plt.plot(cummulative_age)


# # - Pclass,Sex,Age

# In[153]:


f,ax = plt.subplots(1,2,figsize = (18,8))
sns.violinplot("Pclass","Age",hue = "Survived",data = df_train,ax = ax[0],scale = 'count',split= True)
sns.violinplot("Sex","Age",hue = "Survived",data = df_train,ax = ax[1],scale = "count",split = True)


# # -Embarked

# In[154]:


f ,ax = plt.subplots(1,1,figsize = (8,8))
df_train[["Embarked","Survived"]].groupby(["Embarked"]).mean().sort_values(by = "Survived",ascending = False).plot.bar(ax = ax)


# In[155]:


# feature 별 embarked

f,ax = plt.subplots(2,2,figsize = (20,15))
sns.countplot("Embarked",data = df_train,ax = ax[0,0])
sns.countplot("Embarked",hue = "Sex",data = df_train,ax = ax[0,1])
sns.countplot("Embarked",hue = "Survived",data = df_train,ax = ax[1,0])
sns.countplot("Embarked",hue = "Pclass",data = df_train,ax= ax[1,1])


# # -Family = Sibsp + Parch

# In[156]:


df_train["FamilySize"] = df_train["SibSp"] + df_train["Parch"] + 1
df_test["FamilySize"] = df_test["SibSp"] + df_test["Parch"] + 1


# In[157]:


f,ax = plt.subplots(1,3,figsize = (20,8))
sns.countplot("FamilySize",data = df_train,ax = ax[0])
sns.countplot("FamilySize",hue = "Survived",data = df_train,ax = ax[1])
df_train[["FamilySize","Survived"]].groupby(["FamilySize"]).mean().sort_values(by = "Survived",ascending = False).plot.bar(ax = ax[2])


# # -Fare

# In[158]:


fig,ax = plt.subplots(1,1,figsize = (8,8))
g = sns.distplot(df_train["Fare"],label = "Skewness : {:.2f}".format(df_train["Fare"].skew()),ax = ax)
g = g.legend(loc = "best")


# In[159]:


# high skewness

df_train["Fare"] = df_train["Fare"].map(lambda i : np.log(i) if i > 0 else 0)
df_test["Fare"] = df_test["Fare"].map(lambda i : np.log(i) if i > 0 else 0)


# In[160]:


plt.figure(figsize = (8,8))
g = sns.distplot(df_train["Fare"],label = "skewness = {:.2f}".format(df_train["Fare"].skew()))
g = g.legend(loc = "best")


# # Fill NULL

# In[161]:


df_train["Initial"] = df_train.Name.str.extract('([A-Za-z]+)\.')
df_test["Initial"] = df_train.Name.str.extract('([A-Za-z]+)\.')


# In[162]:


pd.crosstab(df_train["Sex"],df_train["Initial"])


# In[163]:


df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)


# In[164]:


df_train.groupby("Initial").mean()


# In[165]:


df_train.groupby('Initial')["Survived"].mean().plot.bar()


# In[166]:


df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == 'Mr'),'Age'] =33
df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == "Mrs"),'Age'] = 36
df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Master'),'Age'] = 5
df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Miss'),'Age'] = 22
df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Other'),'Age'] = 46

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Mr'),'Age'] = 33
df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Mrs'),'Age'] = 36
df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Master'),'Age'] = 5
df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Miss'),'Age'] = 22
df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Other'),'Age'] = 46


# In[167]:


#df_train.loc[df_train.Embarked.isnull(),'Embarked'] = "S"
df_train["Embarked"].fillna("S",inplace = True)


# In[168]:


df_train.Embarked.isnull().any()


# In[169]:


df_test.loc[df_test.Fare.isnull(),'Fare'] = df_test["Fare"].mean()


# In[170]:


# continuous를 categorical 로 바꾸면 자칫 information loss가 생길 수 있음

def category_age(x):
    if x < 10:
        return 0
    elif x < 20:
        return 1
    elif x < 30:
        return 2
    elif x < 40:
        return 3
    elif x < 50:
        return 4
    elif x < 60:
        return 5
    elif x < 70:
        return 6
    else:
        return 7  
    
df_train["Age_cat"] = df_train["Age"].apply(category_age)
df_test["Age_cat"] = df_test["Age"].apply(category_age)


# In[171]:


df_train.drop(["Age"],axis = 1, inplace = True)
df_test.drop(["Age"],axis = 1, inplace = True)


# # String to Numerical
# ## - map으로 1차 -> 공분산 확인
# ## - one-hot  - encoding으로 2차

# In[172]:


df_train["Initial"] = df_train["Initial"].map({"Master" : 0,"Miss": 1,"Mr":2,"Mrs":3,"Other":4})
df_test["Initial"] = df_test["Initial"].map({"Master" : 0,"Miss": 1,"Mr":2,"Mrs":3,"Other":4})


# In[173]:


df_train["Embarked"] = df_train["Embarked"].map({"C":0,"Q":1,"S":2 })
df_test["Embarked"] = df_test["Embarked"].map({"C":0,"Q":1,"S":2 })


# In[174]:


df_train["Sex"] = df_train["Sex"].map({"female":0,"male":1})
df_test["Sex"] = df_test["Sex"].map({"female":0,"male":1})


# In[175]:


heatmap_data = df_train[["Survived","Pclass","Sex","Fare","Embarked","FamilySize","Initial","Age_cat"]]

colormap = plt.cm.RdBu
plt.figure(figsize = (14,12))
sns.heatmap(heatmap_data.astype(float).corr(),annot = True,annot_kws = {"size" : 16})

del heatmap_data


# In[176]:


df_train = pd.get_dummies(df_train,columns = ['Initial'],prefix = 'Initial')
df_test = pd.get_dummies(df_test,columns = ['Initial'],prefix = 'Initial')
df_train = pd.get_dummies(df_train,columns = ['Embarked'],prefix = 'Embarked')
df_test = pd.get_dummies(df_test,columns = ['Embarked'],prefix = 'Embarked')
df_train = pd.get_dummies(df_train,columns = ['Sex'],prefix = 'Sex')
df_test = pd.get_dummies(df_test,columns = ['Sex'],prefix = 'Sex')


# In[178]:


df_train.drop(['PassengerId',"Name","SibSp","Parch","Ticket","Cabin"],axis=1,inplace = True)
df_test.drop(['PassengerId',"Name","SibSp","Parch","Ticket","Cabin"],axis=1,inplace = True)


# In[180]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

# values를 사용해서 dataframe을 numpy로
X_train = df_train.drop("Survived",axis = 1).values
target_label = df_train["Survived"].values
X_test = df_test.values


# In[186]:


X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2022)


# In[187]:


model = RandomForestClassifier()
model.fit(X_tr,y_tr)
prediction = model.predict(X_vld)


# In[188]:


metrics.accuracy_score(prediction,y_vld)


# In[189]:


from pandas import Series

feature_importance = model.feature_importances_
Series_feat_imp = Series(feature_importance,index = df_test.columns)
Series_feat_imp.sort_values(ascending = True).plot.bar()
plt.show()

