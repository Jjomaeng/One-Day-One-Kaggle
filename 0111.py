#!/usr/bin/env python
# coding: utf-8

# In[68]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[69]:


data = pd.read_csv('../data/titanic/train.csv')


# In[70]:


data.isnull().sum()


# In[71]:


f,ax = plt.subplots(1,2,figsize = (18,8))
data['Survived'].value_counts().plot.pie(autopct = '%1.1f%%',ax = ax[0])
sns.countplot("Survived",data = data ,ax = ax[1])
plt.show()


# In[72]:


data.groupby(['Sex',"Survived"])['Survived'].count()


# In[73]:


f,ax = plt.subplots(1,2,figsize = (18,8))
data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax = ax[0])
sns.countplot('Sex',hue = 'Survived',data = data, ax = ax[1])
plt.show()


# In[74]:


pd.crosstab(data.Pclass,data.Survived,margins = True)


# In[75]:


f,ax = plt.subplots(1,2,figsize = (18,8))
data['Pclass'].value_counts().plot.bar(ax = ax[0])
sns.countplot("Pclass",hue = 'Survived',data = data,ax = ax[1])
plt.show()


# In[76]:


pd.crosstab([data.Sex,data.Survived],data.Pclass,margins = True)


# In[77]:


sns.factorplot("Pclass","Survived",hue = "Sex",data = data,size = 6,  aspect = 1.5)


# In[78]:


f,ax = plt.subplots(1,2,figsize = (18,8))
sns.violinplot("Pclass","Age",hue = 'Survived',data = data,split = True,ax = ax[0])
sns.violinplot("Sex","Age",hue = "Survived",data = data,split =True,ax = ax[1])
ax[0].set_yticks(range(0,110,10))
ax[1].set_yticks(range(0,110,10))
plt.show()


# In[79]:


data['Initial'] = data.Name.str.extract('([A-Za-z]+)\.')


# In[80]:


pd.crosstab(data.Sex,data.Initial)


# In[81]:


data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)


# In[82]:


data.groupby('Initial')['Age'].mean()


# In[83]:


data.loc[(data.Age.isnull())&(data.Initial == 'Mr'),'Age'] = 33
data.loc[(data.Age.isnull())&(data.Initial == 'Mrs'),'Age'] = 36
data.loc[(data.Age.isnull())&(data.Initial == 'Master'),'Age'] = 5
data.loc[(data.Age.isnull())&(data.Initial == 'Miss'),'Age'] = 22
data.loc[(data.Age.isnull())&(data.Initial == 'Other'),'Age'] = 46


# In[84]:


data['Age'].isnull().any()


# In[85]:


f,ax = plt.subplots(1,2,figsize = (20,10))
data[data["Survived"] == 0].Age.plot.hist(ax = ax[0],bins = 20,edgecolor = "black")
ax[0].set_title("Survived == 0")


data[data["Survived"]==1].Age.plot.hist(ax = ax[1],bins = 20,edgecolor = 'black')
ax[1].set_title("Survived == 1")


# In[86]:


sns.factorplot('Pclass','Survived',col="Initial",data = data)


# In[87]:


pd.crosstab([data.Embarked,data.Pclass],[data.Sex,data.Survived],margins = True)


# In[88]:


sns.factorplot('Embarked','Survived',data = data)
fig = plt.gcf()
fig.set_size_inches(5,3)
plt.show()


# In[89]:


f,ax = plt.subplots(2,2,figsize = (20,15))
sns.countplot("Embarked",data = data,ax = ax[0,0])
sns.countplot('Embarked',hue = 'Sex',data = data,ax = ax[0,1])
sns.countplot("Embarked",hue= "Survived",data = data,ax = ax[1,0])
sns.countplot("Embarked",hue = 'Pclass',data = data,ax= ax[1,1])


# In[90]:


sns.factorplot("Pclass","Survived",hue='Sex',col ="Embarked",data = data)


# In[91]:


data['Embarked'].fillna('S',inplace = True)
data.Embarked.isnull().any()


# In[92]:


pd.crosstab([data.SibSp],data.Survived)


# In[93]:


sns.barplot('SibSp',"Survived",data = data)
sns.factorplot('SibSp','Survived',data=data,size = 6, aspect = 1.5)
plt.show()


# In[ ]:





# In[94]:


pd.crosstab(data.SibSp,data.Pclass)


# In[95]:


pd.crosstab(data.Parch,data.Pclass)


# In[96]:


sns.barplot('Parch','Survived',data=data)
sns.factorplot('Parch',"Survived",data=data,size = 6, aspect = 1.5)
plt.show()


# In[97]:


f,ax = plt.subplots(1,3,figsize = (20,8))
sns.distplot(data[data["Pclass"]== 1].Fare, ax = ax[0])
sns.distplot(data[data["Pclass"]==2].Fare,ax = ax[1])
sns.distplot(data[data["Pclass"]==3].Fare,ax = ax[2])


# In[98]:


sns.heatmap(data.corr(),annot = True,linewidths = 0.2)
fig.set_size_inches(10,8)
plt.show()


# In[99]:


data["Age_band"] = 0
data.loc[data["Age"]<= 16,"Age_band"] = 0
data.loc[(data["Age"]>16) & (data["Age"]<= 32),"Age_band" ] =1
data.loc[(data["Age"]>32) & (data["Age"]<= 48),"Age_band" ] =2
data.loc[(data["Age"]>48) & (data["Age"]<= 64),"Age_band" ] =3
data.loc[data["Age"]>64,"Age_band"] = 4


# In[100]:


data["Age_band"].value_counts().to_frame()


# In[101]:


sns.factorplot("Age_band","Survived",col = "Pclass",data = data)


# In[102]:


data["Family_Size"] = 0
data["Family_Size"] = data["Parch"] + data["SibSp"]
data["Alone"] = 0
data.loc[(data["Family_Size"]== 0),"Alone"] =1

sns.factorplot("Family_Size","Survived",data = data)
sns.factorplot("Alone","Survived",data = data)


# In[103]:


sns.factorplot("Alone","Survived",col = "Pclass",hue = "Sex",data = data)


# In[104]:


data["Fare_Range"] = pd.qcut(data['Fare'],4)
data.groupby(["Fare_Range"])["Survived"].mean().to_frame()


# In[105]:


data['Fare_cat']=0
data.loc[data['Fare']<=7.91,'Fare_cat']=0
data.loc[(data['Fare']>7.91)&(data['Fare']<=14.454),'Fare_cat']=1
data.loc[(data['Fare']>14.454)&(data['Fare']<=31),'Fare_cat']=2
data.loc[(data['Fare']>31)&(data['Fare']<=513),'Fare_cat']=3


# In[106]:


sns.factorplot('Fare_cat',"Survived",data = data,hue = 'Sex')


# In[107]:


data["Sex"].replace(["male",'female'],[0,1],inplace = True)
data["Embarked"].replace(["S",'C','Q'],[0,1,2],inplace = True)
data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)


# In[108]:


data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)
sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})
fig=plt.gcf()
fig.set_size_inches(18,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# In[109]:


from sklearn.linear_model import LogisticRegression 
from sklearn import svm 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.metrics import confusion_matrix


# In[110]:


train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Survived'])
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]
X=data[data.columns[1:]]
Y=data['Survived']


# In[111]:


model = svm.SVC(kernel = 'rbf',C = 1,gamma = 0.1) #c는 오류 허용 정도
model.fit(train_X,train_Y)
prediction1=model.predict(test_X)
print(metrics.accuracy_score(prediction1,test_Y))


# In[112]:


model = svm.SVC(kernel = 'linear',C = 0.1,gamma = 0.1) 
model.fit(train_X,train_Y)
prediction2=model.predict(test_X)
print(metrics.accuracy_score(prediction2,test_Y))


# In[113]:


model = LogisticRegression()
model.fit(train_X,train_Y)
prediction3 = model.predict(test_X)
print(metrics.accuracy_score(prediction3,test_Y))


# In[119]:


model = DecisionTreeClassifier()
model.fit(train_X,train_Y)
prediction4 = model.predict(test_X)
print(metrics.accuracy_score(prediction4,test_Y))


# In[120]:


model = KNeighborsClassifier()
model.fit(train_X,train_Y)
prediction5 = model.predict(test_X)
print(metrics.accuracy_score(prediction5,test_Y))


# In[121]:


#knn의 n찾기
a_index = list(range(1,11))
a = pd.Series()
x = [i for i in range(0,10+1)]
for i in list(range(1,11)):
    model = KNeighborsClassifier(n_neighbors = i)
    model.fit(train_X,train_Y)
    prediction = model.predict(test_X)
    a = a.append(pd.Series(metrics.accuracy_score(prediction,test_Y)))
plt.plot(a_index,a)
plt.show()
print(a.values.max())


# In[122]:


model = GaussianNB()
model.fit(train_X,train_Y)
prediction6 = model.predict(test_X)
print(metrics.accuracy_score(prediction6,test_Y))


# In[123]:


model = RandomForestClassifier(n_estimators=100) # 생성할 트리의 개수 100
model.fit(train_X,train_Y)
prediction7 = model.predict(test_X)
print(metrics.accuracy_score(prediction7,test_Y))


# In[125]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
kfold = KFold(n_splits=10)
xyz = []
accuracy = []
std = []
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]
for i in models :
    model = i
    cv_result = cross_val_score(model,X,Y,cv = kfold,scoring = "accuracy")
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2 = pd.DataFrame({'CV Mean':xyz,"Std":std},index = classifiers)
new_models_dataframe2


# In[127]:


plt.subplots(figsize = (12,6))
box = pd.DataFrame(accuracy,index=[classifiers])
box.T.boxplot()


# In[129]:


new_models_dataframe2["CV Mean"].plot.barh(width = 0.8)


# In[130]:


f,ax=plt.subplots(3,3,figsize=(12,10))
y_pred = cross_val_predict(svm.SVC(kernel='rbf'),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')
ax[0,0].set_title('Matrix for rbf-SVM')
y_pred = cross_val_predict(svm.SVC(kernel='linear'),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,1],annot=True,fmt='2.0f')
ax[0,1].set_title('Matrix for Linear-SVM')
y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,2],annot=True,fmt='2.0f')
ax[0,2].set_title('Matrix for KNN')
y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,0],annot=True,fmt='2.0f')
ax[1,0].set_title('Matrix for Random-Forests')
y_pred = cross_val_predict(LogisticRegression(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,1],annot=True,fmt='2.0f')
ax[1,1].set_title('Matrix for Logistic Regression')
y_pred = cross_val_predict(DecisionTreeClassifier(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,2],annot=True,fmt='2.0f')
ax[1,2].set_title('Matrix for Decision Tree')
y_pred = cross_val_predict(GaussianNB(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[2,0],annot=True,fmt='2.0f')
ax[2,0].set_title('Matrix for Naive Bayes')
plt.subplots_adjust(hspace=0.2,wspace=0.2)
plt.show()


# In[131]:


from sklearn.model_selection import GridSearchCV
C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['rbf','linear']
hyper={'kernel':kernel,'C':C,'gamma':gamma}
gd=GridSearchCV(estimator=svm.SVC(),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)


# In[132]:


n_estimators=range(100,1000,100)
hyper={'n_estimators':n_estimators}
gd=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)


# In[133]:


from sklearn.ensemble import VotingClassifier
ensemble_lin_rbf=VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=10)),
                                              ('RBF',svm.SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),
                                              ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),
                                              ('LR',LogisticRegression(C=0.05)),
                                              ('DT',DecisionTreeClassifier(random_state=0)),
                                              ('NB',GaussianNB()),
                                              ('svm',svm.SVC(kernel='linear',probability=True))
                                             ], 
                       voting='soft').fit(train_X,train_Y)
print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(test_X,test_Y))
cross=cross_val_score(ensemble_lin_rbf,X,Y, cv = 10,scoring = "accuracy")
print('The cross validated score is',cross.mean())


# In[134]:


from sklearn.ensemble import BaggingClassifier
model=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators=700)
model.fit(train_X,train_Y)
prediction=model.predict(test_X)
print('The accuracy for bagged KNN is:',metrics.accuracy_score(prediction,test_Y))
result=cross_val_score(model,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for bagged KNN is:',result.mean())


# In[135]:


model=BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=0,n_estimators=100)
model.fit(train_X,train_Y)
prediction=model.predict(test_X)
print('The accuracy for bagged Decision Tree is:',metrics.accuracy_score(prediction,test_Y))
result=cross_val_score(model,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for bagged Decision Tree is:',result.mean())


# In[136]:


from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)
result=cross_val_score(ada,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for AdaBoost is:',result.mean())


# In[137]:


from sklearn.ensemble import GradientBoostingClassifier
grad=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)
result=cross_val_score(grad,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for Gradient Boosting is:',result.mean())


# In[140]:


n_estimators=list(range(100,1100,100))
learn_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
hyper={'n_estimators':n_estimators,'learning_rate':learn_rate}
gd=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)


# In[141]:


ada = AdaBoostClassifierClassifierClassifierrandom_state=stClassifier(n_estimators = 200,random_state = 0, learning_rate = 0.05)
result = cross_val_predict(ada,X,Y,cv = 10)
sns.heatmap(confusion_matrix(Y,result),annot = True)
plt.show()


# In[142]:


f,ax=plt.subplots(2,2,figsize=(15,12))
model=RandomForestClassifier(n_estimators=500,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,0])
ax[0,0].set_title('Feature Importance in Random Forests')

model=AdaBoostClassifier(n_estimators=200,learning_rate=0.05,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,1],color='#ddff11')
ax[0,1].set_title('Feature Importance in AdaBoost')

model=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,0],cmap='RdYlGn_r')
ax[1,0].set_title('Feature Importance in Gradient Boosting')

plt.show()

