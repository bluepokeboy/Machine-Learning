# Load libraries
import numpy as np
import pandas
from sklearn import model_selection

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# Split-out validation dataset
array=dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Initialise weights
numFeat=X_train.shape[1]
w=np.zeros((1, numFeat))
b=0
i=0

# Def
def sigmoid(num):
    out=1/(1+np.exp(-num))
    return out

def costFun(w, b, X, Y):
    numEx=X.shape[0]
    new=sigmoid(np.dot(w, X.T)+b)
    cost=(-1/m)*(np.sum((-Y.T*np.log(new))+((1-Y.T)*(np.log(1-new)))))
    return cost

def findGrad(w, b, X, Y):
    numEx=X.shape[0]
    new=sigmoid(np.dot(w, X.T)+b)
    dw = (1/m)*(np.dot(X.T, (new-Y.T).T))
    db = (1/m)*(np.sum(new-Y.T))
    grads={'dw':dw, 'db':db}
    return grads

def update(w, b, X, Y, lr):
    grads=findGrad(w, b, X, Y)
    dw=grads['dw']
    db=grads['db']
    w=w-lr*dw
    b=b-lr*db
    prop={'w':w, 'b':b, 'dw':dw, 'db':db}
    return prop

def findWeight(w, b, X, Y, lr):
    MOE=0.00001
    grads=findGrad(w, b, X, Y)
    while dw>MOE or db>MOE:
        i=i+1
        if i%100 == 0:
            cost=costFun(w, b, X_train, Y_train)
            print(cost)
        prop=update(w, b, X, Y, lr)
        w=prop['w']
        b=prop['b']
        dw=prop['dw']
        db=prop['db']
    finalWeights={'w':w, 'b':b}
    return finalWeights

finalWeights=findWeight(w, b, X_train, Y_train, 0.01)
finW=finalWeights['w']
finB=finalWeights['b']

