#Import Libraries
import numpy as np
import pandas as pd

#Load Data
data=pd.read_csv("Social_Network_Ads.csv")
data=data.sort_values(data.columns[1])
data=data.head(200)
dataset=data.values
table=dataset[:,2:]
#print(table)
X=table[:,:-1]
bias=np.ones((200,1))
X=np.concatenate((bias,X),axis=1)
Y=table[:,-1]

#Initalise Weights
weights=np.zeros(3)

#Find Weights
sdw=1
while sdw>0.001:
    z = np.dot(weights, X.T)
    a=1/(1+np.exp(-z.astype(float)))
    #print(a)
    grad=np.dot(X.T, (a-Y))/200
    weights=weights-0.001*grad
    sdw=sum(abs(grad))
    print(sdw)
#print(weights)

#Predict
age=input('Enter Age')
wage=input('Enter Wage')
X=[1, age, wage]
z=np.dot(weights, X.T)
a=1/(1+np.exp(-z.astype(float)))
if a >= 0.5:
    print('yes')
else:
    print('no')

