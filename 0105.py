#!/usr/bin/env python
# coding: utf-8

# In[120]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[121]:


df_train = pd.read_csv("../data/titanic/train.csv")
df_test = pd.read_csv("../data/titanic/test.csv")


# In[122]:


df_train.describe()


# In[123]:


df_train.info()


# In[124]:


for col in df_train.columns:
    msg = "{:<10}\t:{:.2f}%".format(col,100* df_train[col].isnull().sum()/df_train[col].shape[0])
    print(msg)


# In[125]:


for col in df_test.columns:
    msg = "{:<10}\t:{:.2f}%".format(col,100* df_test[col].isnull().sum()/df_test[col].shape[0])
    print(msg)


# In[126]:


msno.matrix(df = df_train,figsize = (8,8))


# In[127]:


msno.bar(df= df_train,figsize = (8,8))


# In[128]:


f,ax = plt.subplots(1,2,figsize = (16,8))
df_train["Survived"].value_counts().plot.pie(autopct = "%1.1f%%" ,ax = ax[0])
sns.countplot("Survived",data = df_train,ax = ax[1])


# In[129]:


pd.crosstab(df_train["Pclass"],df_train["Survived"],margins = True)


# In[130]:


f,ax = plt.subplots(1,2,figsize = (16,8))
sns.countplot("Pclass",data = df_train,ax = ax[0])
df_train[["Pclass","Survived"]].groupby(['Pclass']).mean().sort_values(by = "Survived",ascending = False).plot.bar(ax = ax[1])


# In[131]:


pd.crosstab(df_train["Sex"],df_train["Survived"],margins = True)


# In[132]:


f,ax = plt.subplots(1,2,figsize = (16,8))
sns.countplot("Sex",data = df_train,ax = ax[0])
df_train[["Sex","Survived"]].groupby(["Sex"]).mean().sort_values(by = "Survived",ascending = False).plot.bar(ax = ax[1])


# In[133]:


plt.figure(figsize = (8,8))
sns.factorplot("Pclass","Survived",hue = "Sex",data = df_train,size = 6, aspect = 1.5)
plt.show()


# In[134]:


sns.factorplot("Sex","Survived",col = "Pclass",data = df_train)


# In[135]:


plt.figure(figsize = (8,8))
g =sns.distplot(df_train["Age"],label = "skewness : {:.2f}".format(df_train["Age"].skew()))
g = g.legend(loc = "best")


# In[136]:


f,ax = plt.subplots(1,1,figsize = (8,8))
sns.kdeplot(df_train[df_train["Survived"]==1]["Age"],ax = ax)
sns.kdeplot(df_train[df_train["Survived"]==0]["Age"],ax = ax)
f.legend(['Survived == 1','Survived == 0'])
f.show()


# In[137]:


f,ax = plt.subplots(1,1,figsize = (8,8))
sns.kdeplot(df_train[df_train["Pclass"] == 1]["Age"])
sns.kdeplot(df_train[df_train["Pclass"] == 2]["Age"])
sns.kdeplot(df_train[df_train["Pclass"] == 3]["Age"])
f.legend(["Pclass == 1","Pclass == 2","Pclass == 3"])
plt.show()


# In[138]:


#cummulative_survival_rate
rate = []
for i in range(0,80):
    rate.append(df_train[df_train["Age"] < i]["Survived"].sum()/len(df_train[df_train["Age"]<i]))
plt.plot(rate)


# In[139]:


f,ax = plt.subplots(1,2,figsize = (16,8))
sns.violinplot("Sex","Age",hue = "Survived",data = df_train,ax = ax[0],split = True)
sns.violinplot("Pclass","Age",hue = "Survived",data = df_train,ax = ax[1],split = True)


# In[140]:


pd.crosstab(df_train["Embarked"],df_train["Survived"],margins = True)


# In[141]:


f,ax = plt.subplots(2,2,figsize = (20,15))
sns.countplot("Embarked",data = df_train,ax = ax[0,0])
df_train[["Embarked","Survived"]].groupby(["Embarked"]).mean().sort_values(by = 'Survived',ascending = False).plot.bar(ax=ax[0,1])
sns.countplot("Embarked",hue = "Pclass",data = df_train,ax =ax[1,0])
sns.countplot("Embarked",hue = "Sex",data = df_train,ax= ax[1,1])


# In[142]:


sns.violinplot("Embarked","Age",hue = "Survived",data = df_train,split = True)


# In[143]:


df_train["FamilySize"] = df_train["SibSp"] + df_train["Parch"] +1
df_test["FamilySize"] = df_test["SibSp"] + df_test["Parch"] +1


# In[144]:


f,ax = plt.subplots(1,3,figsize = (18,6))
sns.countplot("FamilySize",data = df_train,ax = ax[0])
sns.countplot("FamilySize",hue = "Survived",data = df_train,ax =ax[1])
df_train[["FamilySize","Survived"]].groupby(["FamilySize"]).mean().sort_values(by = "Survived",ascending = False).plot.bar(ax = ax[2])


# In[145]:


g = sns.distplot(df_train["Fare"],label = "Skewness = {:.2f}".format(df_train["Fare"].skew()))
g = g.legend(loc = "best")


# In[146]:


df_train["Fare"] = df_train["Fare"].map(lambda i : np.log(i) if i > 0 else 0)
df_test["Fare"] = df_train["Fare"].map(lambda i : np.log(i) if i > 0 else 0)


# In[147]:


g = sns.distplot(df_train["Fare"],label = "Skewness = {:.2f}".format(df_train["Fare"].skew()))
g = g.legend(loc = "best")


# In[148]:


# train = age, embarked
# test = age, fare
df_train["Embarked"].fillna("S",inplace = True)
df_train["Embarked"].isnull().any()


# In[149]:


df_test["Fare"].fillna(df_test["Fare"].mean(),inplace=True)
df_test["Fare"].isnull().any()


# In[150]:


df_train["Initial"] = df_train.Name.str.extract('([A-Za-z]+)\.')
df_test["Initial"] = df_test.Name.str.extract('([A-Za-z]+)\.')


# In[151]:


pd.crosstab(df_train["Initial"],df_train["Sex"])


# In[152]:


df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)


# In[153]:


df_train[["Initial","Survived"]].groupby(["Initial"]).mean().sort_values(by = "Survived",ascending = False).plot.bar()


# In[154]:


df_train[["Age","Initial"]].groupby("Initial").mean()


# In[155]:


df_train["Age"] = df_train[df_train["Age"].isnull() & df_train["Initial"] == "Mr"]["Age"] =33
df_train["Age"] = df_train[df_train["Age"].isnull() & df_train["Initial"] == "Master"]["Age"] =5
df_train["Age"] = df_train[df_train["Age"].isnull() & df_train["Initial"] == "Miss"]["Age"] =22
df_train["Age"] = df_train[df_train["Age"].isnull() & df_train["Initial"] == "Mrs"]["Age"] =36
df_train["Age"] = df_train[df_train["Age"].isnull() & df_train["Initial"] == "Other"]["Age"] =46

df_test["Age"] = df_test[df_test["Age"].isnull() & df_test["Initial"] == "Mr"]["Age"] =33
df_test["Age"] = df_test[df_test["Age"].isnull() & df_test["Initial"] == "Master"]["Age"] =5
df_test["Age"] = df_test[df_test["Age"].isnull() & df_test["Initial"] == "Miss"]["Age"] =22
df_test["Age"] = df_test[df_test["Age"].isnull() & df_test["Initial"] == "Mrs"]["Age"] =36
df_test["Age"] = df_test[df_test["Age"].isnull() & df_test["Initial"] == "Other"]["Age"] =46


# In[156]:


# mapping -> covariance -> one -hot -encoding
# Sex,Embakred,age,Initial

df_train["Sex"] = df_train["Sex"].map({"female":0,"male":1})
df_test["Sex"] = df_test["Sex"].map({"female":0,"male":1})

df_train["Embarked"] = df_train["Embarked"].map({"S":0,"Q":1,"C":2})
df_test["Embarked"] = df_test["Embarked"].map({"S":0,"Q":1,"C":2})

df_train['Initial'] = df_train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
df_test['Initial'] = df_test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})


# In[157]:


df_train.head()


# In[158]:


def age_cat(x):
    if x <10 :
        return 0
    elif x<20 :
        return 1
    elif x<30:
        return 2
    elif x<40:
        return 3
    elif x<50:
        return 4
    elif x<60:
        return 5
    elif x<70:
        return 6
    else :
        return 7

df_train["Age"] = df_train["Age"].apply(age_cat)
df_test["Age"] = df_test["Age"].apply(age_cat)


# In[159]:


heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Initial', 'Age']] 

colormap = plt.cm.RdBu
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,
           square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})

del heatmap_data


# In[161]:


df_train = pd.get_dummies(df_train,columns = ["Initial"],prefix = "Initial")
df_test = pd.get_dummies(df_test,columns = ["Initial"],prefix = "Initial")

df_train = pd.get_dummies(df_train,columns = ["Embarked"],prefix = "Embarked")
df_test = pd.get_dummies(df_test,columns = ["Embarked"],prefix = "Embarked")


# In[162]:


df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)


# In[163]:


from sklearn.ensemble import RandomForestClassifier # 유명한 randomforestclassfier 입니다. 
from sklearn import metrics # 모델의 평가를 위해서 씁니다
from sklearn.model_selection import train_test_split


# In[164]:


X_train = df_train.drop('Survived', axis=1).values
target_label = df_train['Survived'].values
X_test = df_test.values


# In[166]:


X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2022)


# In[167]:


model = RandomForestClassifier()
model.fit(X_tr, y_tr)
prediction = model.predict(X_vld)


# In[168]:


print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))


# In[169]:


from pandas import Series

feature_importance = model.feature_importances_
Series_feat_imp = Series(feature_importance, index=df_test.columns)


# In[170]:


plt.figure(figsize=(8, 8))
Series_feat_imp.sort_values(ascending=True).plot.barh()
plt.xlabel('Feature importance')
plt.ylabel('Feature')
plt.show()

