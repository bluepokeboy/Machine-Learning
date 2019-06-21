import numpy as np
import matplotlib as plt
def linearregression(x,y):
    b1=0
    b2=0
    a=0.001
    #d1=np.sum(np.multiply((((b1*x)+b2)-y),x))
    d1=np.sum(np.multiply(((b1*x+b2)-y),x))
    d2=np.sum(((b1*x)+b2)-y)
    #print(d1,d2)
    b1new=b1-a*d1
    b2new=b2-a*d2
    marginoferror=0.00000000001
    #print(b1new,b2new)
    while(abs(d1)>marginoferror and abs(d2)>marginoferror):
        b1=b1new
        b2=b2new
        d1=np.sum(np.multiply(((b1*x+b2)-y),x))
        d2=np.sum(((b1*x)+b2)-y)
        print(d1,d2)
        b1new=b1-a*d1
        b2new=b2-a*d2
    return(b1,b2)

def plot_regression_line(x, y, b): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "m", 
               marker = "o", s = 30) 
  
    # predicted response vector 
    y_pred = b[0] + b[1]*x 
  
    # plotting the regression line 
    plt.plot(x, y_pred, color = "g") 
  
    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 
  
    # function to show plot 
    plt.show() 

def main():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
    # b1=0
    # b2=0
    # a=0.25
    # print(np.sum(np.multiply((b1*x+a),x)))
    answer=linearregression(x,y)
    #print(answer)
    plot_regression_line(x,y,answer)

if __name__== "__main__":
    main()
