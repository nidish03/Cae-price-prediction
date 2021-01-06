import numpy as np
import pandas as pd
train=pd.read_csv("C:\\Users\\user\\Downloads\\train-data.csv")
train = train.drop('Unnamed: 0', axis = 1)
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

sns.pairplot(train)
plt.show()
sns.relplot(x='Price',y='Year',hue='Fuel_Type',data=train)
plt.show()
sns.distplot(train['Kilometers_Driven'])
plt.show()

data=train.copy()
data.dropna()

Kilometers_Driven_1= data.groupby('Kilometers_Driven')['Kilometers_Driven'].agg('count').sort_values(ascending=False)
Kilometers_Driven_1
Kilometers_Driven_2 = Kilometers_Driven_1[Kilometers_Driven_1<=1]
Kilometers_Driven_2
data.Kilometers_Driven=data.Kilometers_Driven.apply(lambda x:'other' if x in Kilometers_Driven_2 else x)

data['Kmpl']=data['Mileage'].apply(lambda x: float(x.split(' ')[0]))
data['CC']=data['Engine'].apply(lambda x: int(x.split(' ')[0]))
data['bhp']=data['Power'].apply(lambda x: x.split(' ')[0])
print(data['bhp'])

columns=['Fuel_Type','Transmission','Owner_Type']
def categories(multi_columns):
    final=data
    i=0
    for field in multi_columns:
        
        print(field)
        data_1=pd.get_dummies(data[field],drop_first=True)
        data.drop([field],axis=1,inplace=True)
        if i == 0:
            final=data_1.copy()
        else:
            final=pd.concat([final,data_1],axis=1)
        i=i+1
    final=pd.concat([data,final],axis=1)
    
    return final
final_data_1=categories(columns)

final_data=final_data_1.drop(columns=["Name","Location","New_Price",'Mileage','Engine','Power','Kilometers_Driven','Year','bhp'],axis=1)
final_data.info()
X=final_data.drop(['Price'],axis=1)
Y=final_data['Price']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 25)
X_train

reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)
y_pred= reg_rf.predict(X_test)
score_1=r2_score(y_test,y_pred)
print("Accuracy on Traing set: ",reg_rf.score(X_train,y_train))
print("Accuracy on Testing set: ",reg_rf.score(X_test,y_test))
print("R2 score", score_1)
