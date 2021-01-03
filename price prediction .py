#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
train=pd.read_csv("C:\\Users\\user\\Downloads\\train-data.csv")
train = train.drop('Unnamed: 0', axis = 1)


# In[4]:


train


# In[15]:


print(train.info())
print(train.describe())

import matplotlib.pyplot as plt
import seaborn as sns

fig = sns.countplot(x='Power', data=train, palette='Set2')
plt.show()


train['Mileage'].fillna(train['Mileage'].mode()[0],inplace=True)
train['Engine'].fillna(train['Engine'].mode()[0],inplace=True)
train['Seats'].fillna(train['Seats'].mode()[0],inplace=True)
train['Power'].fillna(train['Power'].mode()[0],inplace=True)
train['New_Price'].fillna(train['New_Price'].mode()[0],inplace=True)

missing=train.isnull().sum()
print(missing)








# In[16]:


train_1=train.drop(columns=["Name","Location"],axis=1)

train_2 = pd.get_dummies(train_1,drop_first=True)
print(train_2)

X=train_2.drop(['Price'],axis=1)
Y=train_2['Price']
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 25)

from sklearn.preprocessing import StandardScaler
stndrd=StandardScaler()
X_train=stndrd.fit_transform(X_train)
X_test=stndrd.fit_transform(X_test)


# In[11]:


reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)
y_pred= reg_rf.predict(X_test)
score_1=r2_score(y_test,y_pred)
print("Accuracy on Traing set: ",reg_rf.score(X_train,y_train))
print("Accuracy on Testing set: ",reg_rf.score(X_test,y_test))
print("R2 score", score_1)


# In[ ]:




