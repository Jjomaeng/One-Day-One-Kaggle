#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale = 2.5)

import missingno as msno
import warnings
warnings.filterwarnings(action='ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:


df_train = pd.read_csv("../data/titanic/train.csv")
df_test = pd.read_csv("../data/titanic/test.csv")


# In[51]:


df_train.head()


# In[52]:


df_train.describe()


# In[53]:


df_test.describe()


# ## 1. NULL data check

# In[54]:


for col in df_train.columns :
    msg = 'column: {:>10}\t Percent of NaN value : {:.2f} %'.format(col,100*(df_train[col].isnull().sum()/df_train[col].shape[0]))
    print(msg)


# In[55]:


for col in df_test.columns:
    msg = "column: {:>10}\t Percent of NaN value : {:.2f}%".format(col,100 *(df_test[col].isnull().sum()/df_test[col].shape[0]))
    print(msg)


# In[56]:


msno.matrix(df=df_train.iloc[:,:],figsize = (8,8),color = (0.8,0.5,0.2))


# In[57]:


msno.matrix(df=df_test.iloc[:,:],figsize = (8,8))


# In[58]:


msno.bar(df=df_train.iloc[:,:],figsize = (8,8),color = (0.8,0.5,0.2))


# In[59]:


msno.bar(df=df_test.iloc[:,:],figsize = (8,8))


# ## 2.Target label 확인

# In[60]:


f,ax = plt.subplots(1,2,figsize = (18,8))

df_train["Survived"].value_counts().plot.pie(autopct = "%1.1f%%",ax= ax[0])
ax[0].set_title("pie plot - survived")
sns.countplot('Survived',data = df_train,ax=ax[1])
ax[1].set_title("count plot - survived")

plt.show()


# # 3. EDA

# ### pclass

# In[61]:


df_train[["Pclass","Survived"]].groupby(["Pclass"],as_index = True).count()


# In[62]:


df_train[["Pclass","Survived"]].groupby(["Pclass"],as_index = True).sum()


# In[63]:


## 위에 두 과정을 합치는게 crosstab

pd.crosstab(df_train["Pclass"],df_train["Survived"],margins = True)


# In[64]:


# 생존률 구하기

df_train[['Pclass','Survived']].groupby(['Pclass'],as_index = True).mean().sort_values(by = 'Survived',ascending = False).plot.bar()


# In[65]:


f,ax = plt.subplots(1,2,figsize = (18,8))
df_train["Pclass"].value_counts().plot.bar(ax = ax[0])
sns.countplot("Pclass",hue = "Survived",data = df_train,ax = ax[1])
plt.show()


# ### Sex

# In[66]:


f,ax = plt.subplots(1,2,figsize = (18,8))

df_train["Sex"].value_counts().plot.bar(ax = ax[0])
sns.countplot("Sex",hue = "Survived",data = df_train,ax = ax[1])
plt.show()


# In[67]:


pd.crosstab(df_train["Sex"],df_train["Survived"],margins = True)


# In[68]:


df_train[['Sex',"Survived"]].groupby(["Sex"],as_index= True).mean().sort_values(by = "Survived",ascending = False).plot.bar()


# ### Both Sex and Pclass
#  - 2가지 범주형 데이터에 대한 분석

# In[69]:


# 두 가지 범주형 데이터에 관하여 생존이 어떻게 달라지는 지 확인 -> 3 dimension

sns.factorplot("Pclass","Survived",hue = 'Sex',data = df_train,size = 6, aspect = 1.5)
sns.factorplot("Sex","Survived",col = "Pclass",data = df_train,satureation = .5,size = 9,aspect = 1 )
plt.show()


# ### Age

# In[70]:


plt.figure(figsize = (18,8))
sns.kdeplot(df_train[df_train["Survived"] == 1]["Age"])
sns.kdeplot(df_train[df_train["Survived"] == 0]["Age"])
plt.legend(["Survived == 1","Survived == 0"])
plt.show()


# In[71]:


# 클래스 별 나이 분포 ( 범주 - 연속형 )
plt.figure(figsize = (8,6))
df_train["Age"][df_train["Pclass"] == 1].plot(kind = "kde")
df_train["Age"][df_train["Pclass"] == 2].plot(kind = "kde")
df_train["Age"][df_train["Pclass"] == 3].plot(kind = "kde")

plt.xlabel("Age")
plt.legend(["1st Class","2nd Class","3rd Class"])


# In[72]:


# cummulative survival rate

cummulate_survival_ratio = []
min_n = int(df_train["Age"].min())
max_n = int(df_train["Age"].max())
for i in range(min_n,max_n):
    cummulate_survival_ratio.append(df_train[df_train["Age"] < i]["Survived"].sum() / len(df_train[df_train["Age"] < i]["Survived"]))

plt.figure(figsize = (7,7))
plt.plot(cummulate_survival_ratio)
plt.ylabel("Survival rate")
plt.xlabel("Range of Age")
plt.show()


# ### Pclass , Sex, Age
#  - 2개의 범주 ,1개의 연속

# In[73]:


f,ax = plt.subplots(1,2,figsize = (18,8))
sns.violinplot("Pclass","Age",hue = "Survived", data = df_train,scale = 'count',split = True,ax = ax[0])
sns.violinplot("Sex","Age",hue = "Survived",data = df_train,scale = 'count',split = True,ax = ax[1])


# ### Embarked

# In[74]:


fig,ax = plt.subplots(1,1,figsize = (8,8))
df_train[["Embarked","Survived"]].groupby(["Embarked"]).mean().sort_values(by = "Survived",ascending = False).plot.bar(ax = ax)


# In[75]:


f,ax = plt.subplots(2,2,figsize = (20,15))
sns.countplot("Embarked",data = df_train,ax = ax[0,0])
sns.countplot("Embarked",hue = "Sex",data = df_train,ax = ax[0,1])
sns.countplot("Embarked",hue = "Survived", data = df_train, ax = ax[1,0])
sns.countplot("Embarked",hue = "Pclass",data = df_train,ax = ax[1,1])


# ### sibsp + Parch

# In[76]:


df_train['FamilySize'] = df_train["SibSp"] + df_train["Parch"] + 1
df_test["FamilySize"] = df_test["SibSp"] + df_test["Parch"] +1

print(df_train["FamilySize"].max())
print(df_test["FamilySize"].min())


# In[77]:


fx,ax = plt.subplots(1,3,figsize = (40,10))
sns.countplot("FamilySize",data = df_train,ax = ax[0])
sns.countplot("FamilySize",hue = "Survived",data = df_train,ax = ax[1])
df_train[["FamilySize","Survived"]].groupby(["FamilySize"]).mean().sort_values(by = 'Survived',ascending = False).plot.bar(ax = ax[2])


# ### Fare

# In[78]:


fig,ax = plt.subplots(1,1,figsize = (8,8))
g = sns.distplot(df_train["Fare"],label = "Skewness : {:.2f}".format(df_train["Fare"].skew()),ax = ax)
g = g.legend(loc = "best")


# In[79]:


df_train["Fare"] = df_train["Fare"].map(lambda i : np.log(i) if i > 0 else 0)
df_test["Fare"] = df_test["Fare"].map(lambda i : np.log(i) if i >0 else 0)

plt.figure(figsize = (8,8))
g = sns.distplot(df_train["Fare"],label = 'Skewness : {:.2f}'.format(df_train["Fare"].skew()))
g = g.legend(loc = "best")


# ### ticket

# In[81]:


df_train["Ticket"].value_counts()


# ## Fill NULL

# ### Age

# In[82]:


df_train['Initial']= df_train.Name.str.extract('([A-Za-z]+)\.') 
df_test['Initial']= df_test.Name.str.extract('([A-Za-z]+)\.') 


# In[84]:


pd.crosstab(df_train["Initial"],df_train["Sex"],margins = True)


# In[85]:


df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)


# In[87]:


df_train.groupby("Initial").mean()


# In[88]:


df_train.groupby('Initial')["Survived"].mean().plot.bar()


# In[90]:


df_train.groupby('Initial')["Age"].mean()


# In[92]:


df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == "Mr"),"Age"] = 33
df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == "Mrs"),"Age"] = 36
df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == "Master"),"Age"] = 5
df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == "Miss"),"Age"] = 22
df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == "Other"),"Age"] = 46

df_test.loc[(df_test.Age.isnull()) & (df_test.Initial == "Mr"),"Age"] = 33
df_test.loc[(df_test.Age.isnull()) & (df_test.Initial == "Mrs"),"Age"] = 36
df_test.loc[(df_test.Age.isnull()) & (df_test.Initial == "Master"),"Age"] = 5
df_test.loc[(df_test.Age.isnull()) & (df_test.Initial == "Miss"),"Age"] = 22
df_test.loc[(df_test.Age.isnull()) & (df_test.Initial == "Other"),"Age"] = 46


# In[93]:


sum(df_train['Embarked'].isnull())


# In[94]:


df_train["Embarked"].fillna("S",inplace = True)


# In[97]:


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
    
df_train['Age_cat'] = df_train['Age'].apply(category_age)


# In[99]:


df_train.drop(['Age', 'Age_cat_2'], axis=1, inplace=True)
df_test.drop(['Age'], axis=1, inplace=True)


# In[100]:


df_train["Initial"] = df_train['Initial'].map({"Master" : 0,"Miss" : 1,"Mr":2,"Mrs":3,"Other":4})
df_test["Initial"] = df_test['Initial'].map({"Master" : 0,"Miss" : 1,"Mr":2,"Mrs":3,"Other":4})


# In[101]:


df_train["Embarked"].unique()


# In[102]:


df_train["Embarked"].value_counts()


# In[103]:


df_train["Embarked"] = df_train["Embarked"].map({'C':0,"Q":1,"S":2})
df_test["Embarked"] = df_test["Embarked"].map({'C':0,"Q":1,"S":2})


# In[104]:


df_train["Embarked"].isnull().any()


# In[105]:


df_train["Sex"] = df_train["Sex"].map({'female':0,'male':1})
df_test['Sex'] = df_test["Sex"].map({'female':0,"male":1})


# In[108]:


heatmap_data = df_train[["Survived","Pclass","Sex","Fare","Embarked","FamilySize","Initial","Age_cat"]]

colormap = plt.cm.RdBu
plt.figure(figsize = (14,12))
sns.heatmap(heatmap_data.astype(float).corr(),annot = True)

del heatmap_data


# In[109]:


# one-hotcoding
df_train = pd.get_dummies(df_train,columns = ["Initial"], prefix = "Initial")
df_test = pd.get_dummies(df_test,columns = ["Initial"], prefix = "Initial")

df_train.head()


# In[111]:


df_train = pd.get_dummies(df_train,columns = ["Embarked"],prefix = "Embarked")
df_test = pd.get_dummies(df_test,columns = ["Embarked"],prefix = "Embarked")


# In[112]:


df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)


# In[113]:


df_train.head()


# # Building machine Learning model and prediction using the trained model

# In[114]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[115]:


X_train = df_train.drop("Survived",axis = 1).values
target_label = df_train["Survived"].values
X_test = df_test.values


# In[116]:


X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2022)


# In[118]:


model = RandomForestClassifier()
model.fit(X_tr,y_tr)
prediction = model.predict(X_vld)


# In[119]:


100*metrics.accuracy_score(prediction,y_vld)


# In[120]:


from pandas import Series

feature_importance = model.feature_importances_
Series_feat_imp = Series(feature_importance,index = df_test.columns)


# In[121]:


plt.figure(figsize = (8,8))
Series_feat_imp.sort_values(ascending = True).plot.barh()
plt.show()


# In[122]:


submission = pd.read_csv("../data/titanic/gender_submission.csv")


# In[123]:


submission.head()


# In[125]:


prediction = model.predict(X_test)
submission["Survived"] = prediction
submission.to_csv("../data/titanic/submission/submission.csv")

