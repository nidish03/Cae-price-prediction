#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
train=pd.read_csv("C:\\Users\\user\\Downloads\\train-data.csv")
train = train.drop('Unnamed: 0', axis = 1)


# In[5]:


train


# In[18]:


print(train.info())
print(train.describe())

import matplotlib.pyplot as plt
import seaborn as sns

correlation=train.corr()
sns.heatmap(correlation,annot=True)
plt.show()

train['Mileage'].fillna(train['Mileage'].mode()[0],inplace=True)
train['Engine'].fillna(train['Engine'].mode()[0],inplace=True)
train['Seats'].fillna(train['Seats'].mode()[0],inplace=True)
train['Power'].fillna(train['Power'].mode()[0],inplace=True)
train['New_Price'].fillna(train['New_Price'].mode()[0],inplace=True)

missing=train.isnull().sum()
print(missing)


# In[44]:


sns.pairplot(train)
plt.show()
sns.relplot(x='Price',y='Year',hue='Fuel_Type',data=train)
plt.show()
sns.distplot(train['Kilometers_Driven'])
plt.show()

train_1=train.drop(columns=["Name","Location","New_Price"],axis=1)
train_2 = pd.get_dummies(train_1,drop_first=True)
print(train_2)

X=train_2.drop(['Price'],axis=1)
Y=train_2['Price']


# In[38]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 25)
print(len(X_train))
print(len(y_test))
print(y_test)
from sklearn.preprocessing import StandardScaler
stndrd=StandardScaler()
X_train=stndrd.fit_transform(X_train)
X_test=stndrd.fit_transform(X_test)


# In[39]:


reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)
y_pred= reg_rf.predict(X_test)
score_1=r2_score(y_test,y_pred)
print("Accuracy on Traing set: ",reg_rf.score(X_train,y_train))
print("Accuracy on Testing set: ",reg_rf.score(X_test,y_test))
print("R2 score", score_1)


# In[ ]:




