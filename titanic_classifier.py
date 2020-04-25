# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 19:24:48 2020

@author: Aaron
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r'C:\Users\Aaron\Documents\GitHub\TitanticClassifier\train.csv')
pd.set_option('max_columns',None)
#determining if survived is the target
y = df['Survived']
#the name of the passenger, their ID, and their ticket number should not affect their chance of survival
X = df.drop(['Survived','PassengerId','Name', 'Ticket','Fare'],axis=1)
#print(X.head())
#check if there is any information from any of the features
#print(X.isna().sum())
# of 891 values, missing 177 for age, 687 for cabin, 2 for embarked
#missing 77% of the values for cabin, will have to remove
X = X.drop(['Cabin'],axis=1)
#next look at filling in values missed Embarked
#print(X['Embarked'].value_counts())
#72% of the time, embarked is S, will fill in missing values with S
X['Embarked'].fillna('S',inplace=True)
#now to fill in the missing ages
# g = sns.FacetGrid(data = X, col='Sex')
# g.map(sns.distplot,'Age')

#sns.pairplot(X)




#average about 35 for both sexes, 124 male missing, 53 female missing

#create dummy features for the string features Sex and Embarked
sex_d = pd.get_dummies(X['Sex'])
embarked_d = pd.get_dummies(X['Embarked'])
X = pd.concat([X,sex_d,embarked_d],axis=1)
X = X.drop(['Sex','Embarked','female','Q'],axis=1)




#tried predicting age, unsuccessful
# print('Age Estimation')
# df_age = X.dropna()
# X_train_age = df_age[['Pclass','SibSp','Parch','male','C','S']]
# y_train_age = df_age['Age']
# #print(y_train_age.min())
# from sklearn.linear_model import LinearRegression
# lin_age = LinearRegression()
# lin_age.fit(X_train_age,y_train_age)
# age_preds = lin_age.predict(X_train_age)
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# print(sqrt(mean_squared_error(y_train_age,age_preds)))
# print(X.head())
# X_na = X[X['Age'].isna()].drop('Age',axis=1)

# print(X_na.head())
# replacement_ages = pd.DataFrame(lin_age.predict(X_na), columns=['Age'])
# #print(replacement_ages.shape)
# X_na.reset_index(inplace=True)
# X_na['Age'] = replacement_ages
# X_na.set_index('index',inplace=True)
# X_na['Age'] = X_na['Age'].apply(lambda x: 1 if x<0 else x)
# #print(X_na)
# X = pd.concat([df_age,X_na])





print(X.head())
X['Age'].fillna(29,inplace=True)
#X['Age'] = X.apply(lambda row: 28 if np.isnan(row['Age']) & (row['male'] == 0) else (30.7 if np.isnan(row['Age']) & (row['male'] == 1) else row['Age'] ),axis=1)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X), columns =X.columns)


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
sgd = SGDClassifier()
print('SGD',cross_val_score(sgd, X, y, cv=3, scoring='accuracy'))


from sklearn.svm import SVC
svc = SVC()
print('SVC',cross_val_score(svc, X, y, cv=3, scoring='accuracy'))

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
print('LR',cross_val_score(lr, X, y, cv=3, scoring='accuracy'))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
print('KNN',cross_val_score(knn, X, y, cv=3, scoring='accuracy'))

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
print('DT',cross_val_score(dt, X, y, cv=3, scoring='accuracy'))



X_test = pd.read_csv(r'C:\Users\Aaron\Documents\GitHub\TitanticClassifier\test.csv')
X_test_id = X_test['PassengerId']

sex_d = pd.get_dummies(X_test['Sex'])
embarked_d = pd.get_dummies(X_test['Embarked'])
X_test = X_test.drop(['Sex','Embarked'],axis=1)
X_test = pd.concat([X_test,sex_d,embarked_d],axis=1)
X_test = X_test.drop(['PassengerId','Name','Ticket','Cabin','Fare','female','Q'],axis=1)
X_test['Age'].fillna(29,inplace=True)
X_test = pd.DataFrame(scaler.transform(X_test), columns =X_test.columns)


svc.fit(X,y)
test_preds = pd.DataFrame(svc.predict(X_test))

preds = pd.concat([X_test_id,test_preds],axis=1)
preds.columns = ['PassengerId','Survived']
preds.reset_index(drop=True, inplace=True)

preds.to_csv('predictions.csv')