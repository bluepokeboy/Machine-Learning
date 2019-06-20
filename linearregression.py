import pandas as pd
import numpy as np
from sklearn import model_selection
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
array=dataset.values
X=array[:,0:4]
Y=array[:,4]
test_size=0.2
seed=7
X_train,X_test,Y_train,Y_test=model_selection(X,Y,test_size=test_size,random state=seed)
