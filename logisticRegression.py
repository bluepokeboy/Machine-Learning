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
for i in range(len(Y)):
    if Y[i] == 'Iris-setosa':
        Y[i]=1
    else:
        Y[i]=0
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Initialise weights
numFeat=X_train.shape[1]
w=np.zeros((1, numFeat))
b=0
i=0

def costFun(w, b, X, Y):
    numEx=X.shape[0]
    new=np.dot(w, X.T)+b
    print(type(new))
    cost=(-1/numEx)*(np.sum((-Y.T*np.log(new))+((1-Y.T)*(np.log(1-new)))))
    return cost

def findGrad(w, b, X, Y):
    numEx=X.shape[0]
#    print(X)
#    print(numEx)
    new=np.dot(w, X.T)+b
#    print(new)
#    print(Y)
#    print(new.shape)
#    print(Y.T.shape)
#    print(np.subtract(new, Y.T))
    dw = (1/numEx)*(np.dot(X.T, (new-Y.T).T))
    db = (1/numEx)*(np.sum(new-Y.T))
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
    dw=grads['dw']
#    print(dw)
    sum_dw=sum(abs(dw))
#    print(type(dw[0]))
    db=grads['db']
#    print(db)
    i=0
    while (sum_dw>MOE or abs(db)>MOE):
        print(i)
        if i%100 == 0:
            cost=costFun(w, b, X_train, Y_train)
            print(cost)
        i=i+1
        prop=update(w, b, X, Y, lr)
        w=prop['w']
        b=prop['b']
        dw=prop['dw']
        print("dw="+str(dw.shape))
        sum_dw=sum(abs(dw))
        print("sum_dw="+str(sum_dw))
        db=prop['db']
    finalWeights={'w':w, 'b':b}
    return finalWeights

finalWeights=findWeight(w, b, X_train, Y_train, 0.01)
finW=finalWeights['w']
finB=finalWeights['b']

