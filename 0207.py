#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

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


# In[2]:


train = pd.read_csv("../data/titanic/train.csv")
test = pd.read_csv("../data/titanic/test.csv")
IDtest = test["PassengerId"]


# In[3]:


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
    


# In[4]:


train.loc[Outliers_to_drop]


# In[5]:


train = train.drop(Outliers_to_drop,axis = 0).reset_index(drop = True)


# In[6]:


train_len = len(train)
dataset = pd.concat(objs = [train,test],axis = 0).reset_index(drop = True)


# In[7]:


dataset = dataset.fillna(np.nan)
dataset.isnull().sum()


# In[8]:


train.info()
train.isnull().sum()


# In[9]:


train.head()


# In[10]:


train.dtypes


# In[11]:


train.describe()


# In[12]:


# It doesn't mean that the other features are not usefull.
# Subpopulations in these features can be correlated with the survival.
# To determine this, we need to explore in detail these features

g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot = True,fmt = ".2f",cmap = "coolwarm")


# In[13]:


g = sns.factorplot(x = 'SibSp', y = 'Survived',data = train, kind = "bar", size = 6, palette = "muted")
g.despine(left = True)
g = g.set_ylabels('survival probability')


# In[14]:


g = sns.factorplot(x = 'Parch', y = 'Survived', data = train, kind = 'bar',size = 6, palette = 'muted')
g.despine(left = True)
g = g.set_ylabels('survival probability')


# In[15]:


g = sns.FacetGrid(train,col = 'Survived')
g = g.map(sns.distplot,'Age')


# In[16]:


g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())],color = 'Red',shade = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())],ax = g, color = "Blue",shade = True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])


# In[17]:


dataset["Fare"].isnull().sum()


# In[18]:


dataset["Fare"] = dataset['Fare'].fillna(dataset["Fare"].median())


# In[19]:


g = sns.distplot(dataset["Fare"],color = 'm',label = "Skewness: %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc = 'best')


# In[20]:


dataset["Fare"] = dataset["Fare"].map(lambda i : np.log(i) if i >0 else 0)


# In[21]:


g = sns.distplot(dataset["Fare"],color = 'b',label = 'Skewness : %.2f'%(dataset["Fare"].skew()))
g = g.legend(loc = 'best')


# In[22]:


g = sns.barplot(x = "Sex", y= 'Survived',data = train)
g = g.set_ylabel("Survival Probability")


# In[23]:


train[['Sex','Survived']].groupby('Sex').mean()


# In[24]:


g = sns.factorplot(x = "Pclass", y = 'Survived',data = train,kind = 'bar',size = 6,palette = 'muted')
g.despine(left = True)
g = g.set_ylabels("Survival Probability")


# In[25]:


g = sns.factorplot(x = "Pclass", y = 'Survived',hue = 'Sex',data = train,kind = 'bar',size = 6,palette = 'muted')
g.despine(left = True)
g = g.set_ylabels("Survival Probability")


# In[26]:


dataset["Embarked"].isnull().sum()


# In[27]:


dataset["Embarked"] = dataset["Embarked"].fillna("S")


# In[28]:


g = sns.factorplot(x = "Embarked", y = 'Survived',data = train,kind = 'bar',size = 6,palette = 'muted')
g.despine(left = True)
g = g.set_ylabels("Survival Probability")


# In[29]:


g = sns.factorplot(x = "Pclass", y = 'Survived',col = "Embarked",data = train,kind = 'bar',size = 6,palette = 'muted')
g.despine(left = True)
g = g.set_ylabels("Survival Probability")


# In[30]:


g = sns.factorplot(y = 'Age',x = "Sex", data = dataset, kind = 'box')
g = sns.factorplot(y = 'Age',x = "Sex", hue = 'Pclass',data = dataset, kind = 'box')   
g = sns.factorplot(y = 'Age',x = "Parch", data = dataset, kind = 'box')    
g = sns.factorplot(y = 'Age',x = "SibSp", data = dataset, kind = 'box')


# In[31]:


dataset["Sex"] = dataset["Sex"].map({"male":0,'female':1})


# In[32]:


g = sns.heatmap(dataset[['Age','Sex','SibSp','Parch','Pclass']].corr(),cmap = "BrBG",annot = True)


# In[33]:


index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

for i in index_NaN_age:
    age_med = dataset["Age"].median()
    age_pred = dataset['Age'][((dataset["SibSp"] == dataset.iloc[i]['SibSp']) &(dataset['Parch'] == dataset.iloc[i]['Parch']) & (dataset["Pclass"] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred):
        dataset["Age"].iloc[i] = age_pred
    else :
        dataset["Age"].iloc[i] = age_med


# In[34]:


g = sns.factorplot(x = "Survived",y = "Age", data = train, kind = 'box')
g = sns.factorplot(x = 'Survived', y= 'Age',data = train,kind = 'violin')


# In[35]:


dataset["Name"].head()


# In[36]:


dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
dataset['Title'].head()


# In[37]:


g = sns.countplot(x = "Title", data = dataset)

# Set a property on an artist object.
g = plt.setp(g.get_xticklabels(),rotation = 45)


# In[38]:


dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)


# In[39]:


g = sns.countplot(dataset["Title"])
g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])


# In[40]:


g = sns.factorplot(x = "Title",y = 'Survived',data = dataset, kind = 'bar')
g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
g = g.set_ylabels("survival probability")


# In[41]:


dataset.drop(labels = ["Name"],axis = 1, inplace = True)


# In[42]:


dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1


# In[43]:


g = sns.factorplot(x = "Fsize", y = "Survived",data = dataset)
g = g.set_ylabels("survival probability")


# In[44]:


dataset["Single"] = dataset["Fsize"].map(lambda s : 1 if s == 1 else 0)
dataset["SmallF"] = dataset["Fsize"].map(lambda s : 2 if s == 2 else 0)
dataset["MedF"] = dataset["Fsize"].map(lambda s : 3 if 3 <= s <= 4 else 0)
dataset["LargeF"] = dataset["Fsize"].map(lambda s : 4 if s >= 5 else 0)


# In[45]:


g = sns.factorplot(x = "Single",y = 'Survived',data = dataset, kind = 'bar')
g = g.set_ylabels("survival probability")
g = sns.factorplot(x = "SmallF",y = 'Survived',data = dataset, kind = 'bar')
g = g.set_ylabels("survival probability")
g = sns.factorplot(x = "MedF",y = 'Survived',data = dataset, kind = 'bar')
g = g.set_ylabels("survival probability")
g = sns.factorplot(x = "LargeF",y = 'Survived',data = dataset, kind = 'bar')
g = g.set_ylabels("survival probability")


# In[46]:


dataset = pd.get_dummies(dataset, columns = ["Title"])
dataset = pd.get_dummies(dataset, columns = ["Embarked"],prefix = 'Em')


# In[47]:


dataset.head()


# In[48]:


dataset["Cabin"].head()


# In[49]:


dataset["Cabin"].describe()


# In[50]:


dataset["Cabin"].isnull().sum()


# In[51]:


dataset["Cabin"][dataset["Cabin"].notnull()].head()


# In[52]:


dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset["Cabin"]])


# In[55]:


g = sns.countplot(dataset["Cabin"],order = ["A","B","C","D","E","F","G","T","X"])


# In[56]:


g = sns.factorplot(y="Survived",x="Cabin",data=dataset,kind="bar",order=['A','B','C','D','E','F','G','T','X'])
g = g.set_ylabels("Survival Probability")


# In[57]:


dataset = pd.get_dummies(dataset,columns = ["Cabin"],prefix = 'Cabin')


# In[58]:


dataset["Ticket"].head()


# In[61]:


# strip([chars]) : 인자로 전달된 문자를 String의 왼쪽과 오른쪽에서 제거
Ticket = []

for i in list(dataset.Ticket):
    if not i.isdigit():
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0])
    else:
        Ticket.append('X')
            
dataset["Ticket"] = Ticket
dataset["Ticket"].head()


# In[62]:


dataset = pd.get_dummies(dataset,columns = ["Ticket"],prefix = "T")


# In[63]:


dataset["Pclass"] = dataset["Pclass"].astype('category')
dataset = pd.get_dummies(dataset,columns = ["Pclass"],prefix = "Pc")


# In[64]:


dataset.drop(labels = ["PassengerId"],axis = 1, inplace = True)


# In[65]:


dataset.head()


# In[ ]:




