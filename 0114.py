#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
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

sns.set(style = "white",context = 'notebook',palette = 'deep')


# # Load and check data

# In[2]:


# load data
train = pd.read_csv("../data/titanic/train.csv")
test = pd.read_csv("../data/titanic/test.csv")
IDtest = test["PassengerId"]


# In[3]:


# outlier detection -Tukey method

def detect_outliers(df,n,features):
    outlier_indices = []
    
    for col in features:
        
        Q1 = np.percentile(df[col],25) # 백분위가 25%인 수 반환
        Q3 = np.percentile(df[col],75) # 백분위가 75%인 수 반환
        
        IQR = Q3 - Q1
        
        outlier_step = 1.5 * IQR
        
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        
        outlier_indices.extend(outlier_list_col) # extend : 리스트 안에 원소 자체를 넣어줌
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n)
    
    return multiple_outliers
        
    
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])  


# In[4]:


train.loc[Outliers_to_drop]


# In[5]:


train = train.drop(Outliers_to_drop,axis = 0).reset_index(drop = True) #인덱스를 다시 처음부터 재배열 


# In[6]:


# joining train and test set
train_len = len(train)
dataset = pd.concat(objs = [train,test],axis = 0).reset_index(drop = True)


# In[7]:


# check for null and missing values

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


# # Feature analysis

# ### - Numerical values

# In[12]:


g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot = True,fmt = ".2f",cmap = "coolwarm")


# In[13]:


g = sns.factorplot(x = "SibSp",y = "Survived",data = train ,kind = "bar",size = 6,palette = "muted")
g.despine(left = True)
g = g.set_ylabels("Survival Probability")


# In[14]:


g = sns.factorplot(x = "Parch",y = "Survived",data = train,kind = "bar",size = 6,palette = "muted")
g.despine(left = True) #Remove the top and right spines from plot
g = g.set_ylabels('Survival probability')


# In[15]:


# 여기서 가져갈 것: 공분산 결과 나이 자체의 값 들은 생존율과 크게 연관이 없었는데 나이를 범주화해서 살펴본 결과, 연관성이 보임
g = sns.FacetGrid(train,col = "Survived")
g = g.map(sns.distplot,"Age")


# In[16]:


g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())],color = "Red",shade = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())],color = "Blue",shade = True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])


# In[17]:


dataset["Fare"].isnull().sum()


# In[18]:


dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())


# In[19]:


g = sns.distplot(dataset["Fare"],color = "m",label = 'Skewness : %.2f'%(dataset["Fare"].skew()))
g = g.legend(loc = 'best')


# In[20]:


dataset["Fare"] = dataset["Fare"].map(lambda i : np.log(i) if i > 0 else 0)


# In[21]:


g = sns.distplot(dataset["Fare"],color = 'b',label = "Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc = "best")


# # Categorical values

# In[22]:


g = sns.barplot(x = "Sex",y = "Survived",data = train)


# In[23]:


train[["Sex","Survived"]].groupby('Sex').mean()


# In[24]:


g = sns.factorplot(x = 'Pclass',y = "Survived",data = train,kind = "bar",size = 6, palette = 'muted')
g.despine(left=True)
g = g.set_ylabels("Survival Probability")


# In[25]:


g = sns.factorplot(x = "Pclass",y = 'Survived',hue = 'Sex',data = train,size = 6,kind = 'bar',palette = 'muted')
g.despine(left=True)
g = g.set_ylabels("Survival probability")


# In[26]:


dataset["Embarked"].isnull().sum()


# In[27]:


dataset['Embarked'] = dataset["Embarked"].fillna("S")


# In[28]:


g = sns.factorplot(x = "Embarked",y = 'Survived',data = train,size = 6, kind = "bar",palette = 'muted')
g.despine(left = True)
g = g.set_ylabels('Survival Probability')


# In[29]:


g = sns.factorplot("Pclass",col = "Embarked",data = train,size = 6,kind = "count",palette = "muted")
g.despine(left = True)
g = g.set_ylabels("Count")


# # Filling missing Values

# In[30]:


# Sex is not informative to predict age

g = sns.factorplot(y = "Age",x = "Sex",data = dataset,kind = 'box')
g = sns.factorplot(y = "Age",x = "Sex",hue = "Pclass",data = dataset,kind = 'box')
g = sns.factorplot(y = "Age",x = "Parch",data = dataset,kind = 'box')
g = sns.factorplot(y = "Age",x= "SibSp",data = dataset,kind = "box")


# In[31]:


#convert Sex into categorical
dataset["Sex"] = dataset["Sex"].map({"male":0,"female":1})


# In[32]:


g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap = "BrBG",annot = True)


# In[33]:


#filling missing value of Age

index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

for i in index_NaN_age:
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][((dataset["SibSp"] == dataset.iloc[i]["SibSp"]) & (dataset["Parch"]) & (dataset["Pclass"] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        dataset["Age"].iloc[i] = age_pred
    else :
        dataset['Age'].iloc[i] = age_med
    


# In[34]:


g = sns.factorplot(x = 'Survived', y = 'Age',data = train, kind = "box")
g = sns.factorplot(x = 'Survived', y = 'Age',data = train, kind = "violin")


# In[35]:


dataset["Name"].head()


# In[36]:


dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]] #strip() : 공백제거
dataset["Title"] = pd.Series(dataset_title)
dataset['Title'].head()


# In[37]:


g = sns.countplot(x = "Title",data = dataset)
g = plt.setp(g.get_xticklabels(),rotation = 45) #글꼴 설정


# In[38]:


# Convert to categorical values Title 
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)


# In[39]:


g = sns.countplot(dataset["Title"])
g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])


# In[40]:


g = sns.factorplot(x = "Title",y = "Survived",data = dataset,kind = "bar")
g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
g = g.set_ylabels("survival probability")


# In[41]:


dataset.drop(labels = ["Name"],axis = 1, inplace = True)


# In[42]:


dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1


# In[43]:


g = sns.factorplot(x = "Fsize",y = 'Survived',data = dataset,size = 6, aspect = 1.5)


# In[44]:


dataset["Single"] = dataset["Fsize"].map(lambda s : 1 if s == 1 else 0)
dataset["SmallF"] = dataset["Fsize"].map(lambda s :1 if s == 2 else 0)
dataset["MedF"] = dataset["Fsize"].map(lambda s : 1 if 3 <= s <= 4 else 0)
dataset["LargeF"] = dataset["Fsize"].map(lambda s : 2 if s >= 5 else 0)


# In[45]:


g = sns.factorplot(x="Single",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="SmallF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="MedF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="LargeF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")


# In[46]:


dataset = pd.get_dummies(dataset,columns=["Title"])
dataset = pd.get_dummies(dataset ,columns = ["Embarked"],prefix = "Em")


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


dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else "X" for i in dataset["Cabin"]])


# In[53]:


g = sns.countplot("Cabin",data = dataset)


# In[54]:


g = sns.factorplot(y = "Survived",x = "Cabin",data = dataset,kind = 'bar')


# In[55]:


dataset = pd.get_dummies(dataset,columns = ["Cabin"],prefix = "Cabin")


# In[56]:


dataset["Ticket"].head()


# In[57]:


Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit():
        Ticket.append(i.replace(".","").replace("/","").strip().split(" ")[0])
    else :
        Ticket.append("X")
dataset["Ticket"] = Ticket
dataset["Ticket"].head()


# In[58]:


dataset = pd.get_dummies(dataset,columns = ["Ticket"],prefix = "T")


# In[59]:


dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset,columns = ["Pclass"],prefix = "Pc")


# In[60]:


dataset.drop(labels=["PassengerId"],axis = 1,inplace = True)


# In[61]:


dataset.head()


# # modeling

# In[65]:


train = dataset[:train_len]
test = dataset[train_len:]
test.drop(labels = ["Survived"],axis = 1,inplace = True)


# In[66]:


train["Survived"] = train["Survived"].astype(int)
Y_train = train["Survived"]
X_train = train.drop(labels = ["Survived"],axis = 1)


# In[67]:


kfold = StratifiedKFold(n_splits = 10)


# In[73]:


random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers : #n_jobs : cpu 코어 수 조정
    cv_results.append(cross_val_score(classifier,X_train,y = Y_train,scoring = 'accuracy',cv=kfold,n_jobs=4))
    
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})


# In[75]:


# orient : orient“v” | “h”, optional
#Orientation of the plot (vertical or horizontal). This is usually inferred based on the type of the input variables, 
#but it can be used to resolve ambiguity when both x and y are numeric or when plotting wide-form data.

# xerr, yerr 파라미터에 길이 할당하여 신뢰구간 표현 -> 더 공부
g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})


# In[ ]:


# SAMME.R uses the probability estimates to update the additive model, while SAMME uses the classifications only.
DTC = DecisionTreeClassifier()
adaDTC = AdaBoostClassifier(DTC,random_state = 7)

