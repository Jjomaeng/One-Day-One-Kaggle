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

# In[62]:


train = dataset[:train_len]
test = dataset[train_len:]
test.drop(labels = ["Survived"],axis = 1,inplace = True)


# In[63]:


train["Survived"] = train["Survived"].astype(int)
Y_train = train["Survived"]
X_train = train.drop(labels = ["Survived"],axis = 1)


# In[64]:


kfold = StratifiedKFold(n_splits = 10)


# In[65]:


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


# In[66]:


# orient : orient“v” | “h”, optional
#Orientation of the plot (vertical or horizontal). This is usually inferred based on the type of the input variables, 
#but it can be used to resolve ambiguity when both x and y are numeric or when plotting wide-form data.

# xerr, yerr 파라미터에 길이 할당하여 신뢰구간 표현 -> 더 공부
g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})


# In[68]:


# SAMME.R uses the probability estimates to update the additive model, while SAMME uses the classifications only.
DTC = DecisionTreeClassifier()
adaDTC = AdaBoostClassifier(DTC,random_state = 7)
ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}
gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv = kfold,scoring = 'accuracy',n_jobs = 4,verbose=1)
# verbose : Controls the verbosity: the higher, the more messages.
gsadaDTC.fit(X_train,Y_train)
ada_best = gsadaDTC.best_estimator_


# In[69]:


gsadaDTC.best_score_


# In[70]:


ExtC = ExtraTreesClassifier()
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}
# bootstrapbool, default=False
#Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.

gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsExtC.fit(X_train,Y_train)

ExtC_best = gsExtC.best_estimator_

# Best score
gsExtC.best_score_


# In[71]:


RFC = RandomForestClassifier()

rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_

gsRFC.best_score_


# In[72]:


GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(X_train,Y_train)

GBC_best = gsGBC.best_estimator_

gsGBC.best_score_


# In[73]:


# Support Vector Classification.
SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsSVMC.fit(X_train,Y_train)

SVMC_best = gsSVMC.best_estimator_

gsSVMC.best_score_


# In[74]:


def plot_learning_curve(estimator,title,X,y,ylim = None,cv = None,n_jobs = -1,train_sizes = np.linspace(.1,1.0,5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim) # ??
    plt.xlabel("Training examples")
    plt.ylabel('Score')
    
    # Determines cross-validated training and test scores for different training set sizes.
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    
    # Fill the area between two horizontal curves.
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[75]:


g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",X_train,Y_train,cv=kfold)


# In[77]:


nrows = ncols = 2
fig,axes = plt.subplots(nrows= nrows,ncols = ncols, sharex = 'all',figsize = (15,15))
names_classifiers = [("AdaBoosting", ada_best),("ExtraTrees",ExtC_best),("RandomForest",RFC_best),("GradientBoosting",GBC_best)]

nclassifier = 0

for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[nclassifier][0]
        classifier = names_classifiers[nclassifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        g = sns.barplot(y=X_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])
        g.set_xlabel("Relative importance",fontsize=12)
        g.set_ylabel("Features",fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title(name + " feature importance")
        nclassifier += 1


# In[78]:


test_Survived_RFC = pd.Series(RFC_best.predict(test), name="RFC")
test_Survived_ExtC = pd.Series(ExtC_best.predict(test), name="ExtC")
test_Survived_SVMC = pd.Series(SVMC_best.predict(test), name="SVC")
test_Survived_AdaC = pd.Series(ada_best.predict(test), name="Ada")
test_Survived_GBC = pd.Series(GBC_best.predict(test), name="GBC")


# Concatenate all classifier results
ensemble_results = pd.concat([test_Survived_RFC,test_Survived_ExtC,test_Survived_AdaC,test_Survived_GBC, test_Survived_SVMC],axis=1)


g= sns.heatmap(ensemble_results.corr(),annot=True)


# In[79]:


votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, Y_train)


# In[80]:


test_Survived = pd.Series(votingC.predict(test), name="Survived")

results = pd.concat([IDtest,test_Survived],axis=1)


# In[81]:


results.to_csv("ensemble.csv",index=False)

