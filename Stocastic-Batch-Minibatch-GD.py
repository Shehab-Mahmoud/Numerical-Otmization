# import libraries
from os import truncate
import numpy as np
import matplotlib.pyplot as plt

# training data
x_points =np.array( [[1,1,2,3,4,5,6,7,8,9,10,11]]).T
y_points = np.array([[1,2,3,1,4,5,6,4,7,10,15,9]]).T

def intialize_params(X):
    theta0 = 0
    theta1 = 0
    m = X.shape[0]
    return theta0 ,theta1,m

# function to calculate hypothisis h(X) 
def hyp(theta0,theta1,X):
    return theta0 + theta1*X

# function to compute Stocastic cost
def compute_cost_stocastic(theta0,theta1,X,Y):
    return (1/2)*((hyp(theta0,theta1,X)-Y))**2

# calculate Stocastic grads
def calculate_grads_stocastic(X,Y,theta0,theta1):
    dtheta0 = hyp(theta0,theta1,X)-Y
    dtheta1 = (hyp(theta0,theta1,X)-Y)*X

    return dtheta0,dtheta1

# update paramaters with gradients.
def update_params(theta0,theta1,learning_rate,dtheta0,dtheta1):
    theta0 = theta0 - learning_rate*dtheta0;
    theta1 = theta1 - learning_rate*dtheta1;
    
    return theta0, theta1

def fit_model(X,Y,learning_rate = 0.001,tolerance = 1e-5,plot_result=True,epochs = 1000):
    theta0, theta1,m = intialize_params(X)

    iters = 0
    while(True):
        indexes = np.random.randint(0, len(X)) # random sample
        
        Xs = np.take(X, indexes)
        ys = np.take(Y, indexes)

        cost_old = compute_cost_stocastic(theta0,theta1,Xs,ys)
        dtheta0 , dtheta1 = calculate_grads_stocastic(Xs,ys,theta0,theta1)
        theta0, theta1 = update_params(theta0,theta1,learning_rate,dtheta0,dtheta1)
        cost_new = compute_cost_stocastic(theta0,theta1,Xs,ys)   
        
        iters +=1
        if((cost_old-cost_new)<=tolerance):
            break
    # plot fitted line w.r.t data
    if(plot_result):
        plt.scatter(x_points,y_points)

        points = np.linspace(0,10,10)
        predicted_line = theta0+theta1*points
        
        plt.plot(points,predicted_line,"orange")
        plt.grid()
        plt.legend(["Linear regression fit","Data"])

    print(iters)
    return theta0,theta1



theta0,theta1 = fit_model(x_points,y_points)
print(theta0,theta1)



plt.show()