#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')


# In[2]:


data = pd.read_csv('../data/titanic/train.csv')


# In[3]:


data.head()


# In[4]:


data.isnull().sum()


# In[5]:


f,ax = plt.subplots(1,2,figsize = (18,8))
data["Survived"].value_counts().plot.pie(autopct = '%1.1f%%',ax = ax[0])
sns.countplot("Survived",data = data,ax = ax[1])


# ### sex

# In[6]:


data.groupby(['Sex','Survived'])['Survived'].count()


# In[7]:


f,ax = plt.subplots(1,2,figsize = (18,8))
data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax = ax[0])
sns.countplot("Sex",hue = "Survived",data = data,ax = ax[1])


# ### Pclass

# In[8]:


pd.crosstab(data.Pclass,data.Survived,margins = True)


# In[9]:


f,ax = plt.subplots(1,2,figsize = (18,8))
sns.countplot("Pclass",data = data,ax = ax[0])
sns.countplot("Pclass",hue = "Survived",data = data,ax = ax[1])


# In[10]:


pd.crosstab([data.Sex,data.Survived],data.Pclass,margins = True)


# In[11]:


sns.factorplot("Pclass","Survived",hue = "Sex",data = data)


# ### Age

# In[12]:


data["Age"].describe()


# In[13]:


f,ax = plt.subplots(1,2,figsize = (18,8))
sns.violinplot("Pclass","Age",hue = "Survived",data = data,ax= ax[0],split = True)
sns.violinplot("Sex","Age",hue = "Survived",data = data, ax = ax[1],split = True)


# In[14]:


data['Initial'] = 0
data['Initial'] = data.Name.str.extract('([A-Za-z]+)\.')


# In[15]:


pd.crosstab(data.Sex,data.Initial)


# In[16]:


data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)


# In[17]:


data.groupby(['Initial'])['Age'].mean()


# In[18]:


data.loc[(data.Age.isnull())&(data.Initial == 'Mr'),"Age"] =33
data.loc[(data.Age.isnull())&(data.Initial == "Mrs"),"Age"] = 36
data.loc[(data.Age.isnull())&(data.Initial=='Master'),'Age']=5
data.loc[(data.Age.isnull())&(data.Initial=='Miss'),'Age']=22
data.loc[(data.Age.isnull())&(data.Initial=='Other'),'Age']=46


# In[19]:


data.Age.isnull().any()


# In[20]:


f,ax = plt.subplots(1,2,figsize = (20,10))
data[data["Survived"]== 0].Age.plot.hist(ax = ax[0],bins = 20,edgecolor = "black")
data[data["Survived"]== 1].Age.plot.hist(ax = ax[1],bins = 20,edgecolor = "black")


# In[21]:


sns.factorplot("Pclass","Survived",col = "Initial",data = data)
plt.show()


# In[22]:


pd.crosstab([data.Embarked,data.Pclass],[data.Sex,data.Survived],margins = True).style.background_gradient(cmap = 'summer_r')


# In[23]:


sns.factorplot('Embarked','Survived',data = data,aspect = 1.5)


# In[24]:


f,ax = plt.subplots(2,2,figsize = (20,10))
sns.countplot("Embarked",data = data,ax = ax[0,0])
sns.countplot("Embarked",hue = "Sex",data = data,ax =ax[0,1])
sns.countplot("Embarked",hue = "Survived",data = data, ax = ax[1,0])
sns.countplot("Embarked",hue = "Pclass",data = data, ax= ax[1,1])


# In[25]:


sns.factorplot("Pclass","Survived",data = data,hue = 'Sex',col = "Embarked")


# In[26]:


data["Embarked"].fillna("S",inplace = True)


# In[27]:


data.Embarked.isnull().any()


# In[28]:


pd.crosstab(data.SibSp,data.Survived).style.background_gradient(cmap = 'summer_r')


# In[29]:


sns.barplot('SibSp',"Survived",data = data)
sns.factorplot("SibSp","Survived",data = data)


# In[30]:


pd.crosstab(data.SibSp,data.Pclass).style.background_gradient(cmap = 'summer_r')


# In[31]:


pd.crosstab(data.Parch,data.Pclass).style.background_gradient(cmap = 'summer_r')


# In[32]:


sns.barplot("Parch","Survived",data = data)
sns.factorplot("Parch","Survived",data = data,aspect = 1.5)


# In[33]:


data["Fare"].describe()


# In[34]:


f,ax = plt.subplots(1,3,figsize = (20,8))
sns.distplot(data[data["Pclass"]==1].Fare,ax = ax[0])
sns.distplot(data[data['Pclass']==2].Fare,ax = ax[1])
sns.distplot(data[data["Pclass"]==3].Fare,ax = ax[2])


# In[35]:


sns.heatmap(data.corr(),annot= True,cmap = 'RdYlGn',linewidths = 0.2)
fig = plt.gcf()
fig.set_size_inches(10,8)


# In[36]:


data['Age_band'] = 0
data.loc[data.Age <= 16,'Age_band'] = 0
data.loc[(data.Age > 16) & (data.Age <= 32),"Age_band" ] = 1
data.loc[(data.Age > 32) & (data.Age <= 48),"Age_band"] = 2
data.loc[(data.Age > 48) & (data.Age <= 64),"Age_band"] = 3
data.loc[data.Age >64,'Age_band'] = 4


# In[37]:


data["Age_band"].value_counts().to_frame()


# In[38]:


sns.factorplot("Age_band","Survived",data = data,col = "Pclass")


# In[39]:


data["Family_Size"] = data["SibSp"] + data["Parch"]
data["Alone"] = 0
data.loc[data["Family_Size"] == 0,"Alone"] = 1
sns.factorplot("Family_Size","Survived",data = data,aspect = 1.5)
sns.factorplot("Alone","Survived",data = data,aspect = 1.5)


# In[40]:


sns.factorplot("Alone","Survived",data = data,hue = "Sex",col = "Pclass")


# In[41]:


data["Fare_Range"] = pd.qcut(data["Fare"],4)
data.groupby(["Fare_Range"])["Survived"].mean().to_frame()


# In[42]:


data["Fare_cat"] = 0
data.loc[(data["Fare"]<= 7.91),'Fare_cat'] = 0
data.loc[(data["Fare"]> 7.91) & (data["Fare"]<= 14.454),'Fare_cat'] = 1
data.loc[(data["Fare"]> 14.454) & (data["Fare"]<= 31),'Fare_cat'] = 2
data.loc[(data["Fare"]>31)&(data["Fare"]<= 513),'Fare_cat'] = 3


# In[43]:


sns.factorplot('Fare_cat',"Survived",data = data,hue = "Sex")


# In[44]:


data["Sex"].replace(['male','female'],[0,1],inplace = True)
data["Embarked"].replace(["S","C","Q"],[0,1,2],inplace = True)
data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)


# In[45]:


data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)


# In[46]:


sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})
fig = plt.gcf()
fig.set_size_inches(18,15)


# In[50]:


from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[52]:


train,test = train_test_split(data,test_size = 0.3,random_state = 0,stratify=data["Survived"])
# stratify 옵션 : 분류인 경우 레이블의 클래스 비율을 유지하면서 나누기 때문에 성능에 큰 영향을 줌
train_X = train[train.columns[1:]]
train_Y = train[train.columns[:1]]
test_X = test[test.columns[1:]]
test_Y = test[test.columns[:1]]
X = data[data.columns[1:]]
Y = data["Survived"]


# In[53]:


#svm - 커널 : 가우시안
model = svm.SVC(kernel = "rbf",C=1,gamma = 0.1)
model.fit(train_X,train_Y)
prediction1 = model.predict(test_X)
print(metrics.accuracy_score(prediction1,test_Y))


# In[55]:


#svm - 커널: 선형
model = svm.SVC(kernel = 'linear',C = 0.1, gamma = 0.1)
model.fit(train_X,train_Y)
prediction2 = model.predict(test_X)
print(metrics.accuracy_score(prediction2,test_Y))


# In[57]:


model = LogisticRegression()
model.fit(train_X,train_Y)
prediction3 = model.predict(test_X)
print(metrics.accuracy_score(prediction3,test_Y))


# In[58]:


model = DecisionTreeClassifier()
model.fit(train_X,train_Y)
prediction4 = model.predict(test_X)
print(metrics.accuracy_score(prediction4,test_Y))


# In[64]:


# knn은 default로 n_neighbours = 5
model = KNeighborsClassifier(n_neighbors=9)
model.fit(train_X,train_Y)
prediction5 = model.predict(test_X)
print(metrics.accuracy_score(prediction5,test_Y))


# In[62]:


# 최적의 n 찾기
a_index = list(range(1,11))
a = pd.Series()
x = [i for i in range(1,11)]
for i in list(range(1,11)):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(train_X,train_Y)
    prediction = model.predict(test_X)
    a = a.append(pd.Series(metrics.accuracy_score(prediction,test_Y)))
plt.plot(a_index,a)
plt.show()


# In[65]:


model = GaussianNB()
model.fit(train_X,train_Y)
prediction6 = model.predict(test_X)
print(metrics.accuracy_score(prediction6,test_Y))


# In[66]:


model = RandomForestClassifier(n_estimators = 100)
model.fit(train_X,train_Y)
prediction7 = model.predict(test_X)
print(metrics.accuracy_score(prediction7,test_Y))


# # cross validation

# In[69]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
Kfold = KFold(n_splits = 10,random_state=22,shuffle=True)
xyz = []
accuracy = []
std = []
classifiers = ['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']
models = [svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]
for i in models:
    model = i
    cv_result = cross_val_score(model,X,Y, cv = Kfold,scoring = "accuracy")
    cv_result = cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2 = pd.DataFrame({"CV mean" : xyz,'Std':std},index = classifiers)
new_models_dataframe2


# In[70]:


plt.subplots(figsize = (12,6))
box = pd.DataFrame(accuracy,index = [classifiers])
box.T.boxplot()


# In[72]:


new_models_dataframe2['CV mean'].plot.barh(width = 0.8)
plt.title("Average CV Mean Accuracy")
fig = plt.gcf()
fig.set_size_inches(8,5)
plt.show()


# In[77]:


#cross_val_predict = The data is split according to the cv parameter. Each sample belongs to exactly one test set, 
#                    and its prediction is computed with an estimator fitted on the corresponding training set.
f,ax = plt.subplots(3,3,figsize = (12,10))
y_pred = cross_val_predict(svm.SVC(kernel='rbf'),X,Y,cv = 10)
sns.heatmap(confusion_matrix(Y,y_pred),ax = ax[0,0],annot = True,fmt = '2.0f')
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


# # Hyper_Parameters Tuning

# - SVM

# In[78]:


#GridSearchCV의 iteration시마다 수행 결과 메시지를 출력
#verbose=0(default)면 메시지 출력 안함
#verbose=1이면 간단한 메시지 출력
#verbose=2이면 하이퍼 파라미터별 메시지 출력

from sklearn.model_selection import GridSearchCV
C = [0.05,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
gamma = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel = ['rbf','linear']
hyper = {'kernel':kernel,'C':C,'gamma':gamma}
gd = GridSearchCV(estimator = svm.SVC(),param_grid=hyper,verbose = True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)


# - RandomForest

# In[79]:


n_estimators=range(100,1000,100)
hyper={'n_estimators':n_estimators}
gd=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)


# ## ensemble - Bagging

# In[80]:


from sklearn.ensemble import VotingClassifier
ensemble_lin_rbf = VotingClassifier(estimators = [('KNN',KNeighborsClassifier(n_neighbors=10)),
                                              ('RBF',svm.SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),
                                              ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),
                                              ('LR',LogisticRegression(C=0.05)),
                                              ('DT',DecisionTreeClassifier(random_state=0)),
                                              ('NB',GaussianNB()),
                                              ('svm',svm.SVC(kernel='linear',probability=True))
                                             ], 
                       voting='soft').fit(train_X,train_Y)
# voting : 'hard' = 최종 아웃풋 결과 중 각 모델들이 가장많이 선택한 아웃풋을 최종 아웃풋으로 설정
#          'soft' = 최종 아웃풋 결과의 확률값을 기반으로 평균을 내어, 이중 가장 확률값이 높은 아웃풋을 최종 아웃풋으로 설정

print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(test_X,test_Y))
cross=cross_val_score(ensemble_lin_rbf,X,Y, cv = 10,scoring = "accuracy")
print('The cross validated score is',cross.mean())


# In[81]:


from sklearn.ensemble import BaggingClassifier
model = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators = 700)
model.fit(train_X,train_Y)
prediction = model.predict(test_X)
print(metrics.accuracy_score(prediction,test_Y))
result = cross_val_score(model,X,Y,cv = 10,scoring = "accuracy")
print(result.mean())


# In[82]:


model=BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=0,n_estimators=100)
model.fit(train_X,train_Y)
prediction=model.predict(test_X)
print('The accuracy for bagged Decision Tree is:',metrics.accuracy_score(prediction,test_Y))
result=cross_val_score(model,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for bagged Decision Tree is:',result.mean())


# ## ensemble - Boosting

# In[83]:


from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)
result=cross_val_score(ada,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for AdaBoost is:',result.mean())


# In[84]:


from sklearn.ensemble import GradientBoostingClassifier
grad=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)
result=cross_val_score(grad,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for Gradient Boosting is:',result.mean())


# In[86]:


n_estimators = list(range(100,1100,100))
learn_rate = [0.05,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
hyper = {"n_estimators":n_estimators,'learning_rate':learn_rate}
gd = GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper,verbose = True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)


# In[89]:


ada = AdaBoostClassifier(n_estimators = 100,random_state = 0,learning_rate = 0.1)
result = cross_val_predict(ada,X,Y,cv = 10)
sns.heatmap(confusion_matrix(Y,result),cmap = 'winter',annot = True,fmt = '2.0f')
plt.show()


# In[91]:


f,ax = plt.subplots(2,2,figsize = (15,12))
model = RandomForestClassifier(n_estimators = 500,random_state = 0)
model.fit(train_X,train_Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending = True).plot.barh(width = 0.8,ax = ax[0,0])
ax[0,0].set_title('Feature Importance in Random Forests')
model=AdaBoostClassifier(n_estimators=200,learning_rate=0.05,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,1],color='#ddff11')
ax[0,1].set_title('Feature Importance in AdaBoost')
model=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)
model.fit(X,Y)
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,0],cmap='RdYlGn_r')
ax[1,0].set_title('Feature Importance in Gradient Boosting')

