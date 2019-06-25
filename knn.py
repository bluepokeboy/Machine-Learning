#Import Libs
import numpy as np
import pandas as pd
 

#Collect Data
data=pd.read_csv("weight-height.csv")
#print(data)
table=data.values
dataset=table[:500, 1:]
#print(dataset)

#Input Query
weight=float(input("Enter weight : "))

#print(weight)

#Find Distances from all points
distArr=[]
for index, example in enumerate(dataset):
#    print(type(example[0]))
#    print(type(weight))
    dist=example[0]-weight
    dist=abs(dist)
    distArr.append((dist,index))
#print(distArr)
sortedDistArr=sorted(distArr)
knnArr=sortedDistArr[:20]#took k as 20. Could be further optimized
#print(knnArr)
heightsOfKNN=[dataset[ind][1] for dist, ind in knnArr]
#print(heightsOfKNN)
predHeight=sum(heightsOfKNN)/20
print(predHeight)



