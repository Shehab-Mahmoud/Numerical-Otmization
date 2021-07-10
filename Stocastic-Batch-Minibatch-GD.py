# import libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import array
from sklearn.metrics import r2_score
np.random.seed(2040)

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

# function to caculate batch cost
def compute_cost(theta0,theta1,X,Y,m):
    return (1/(2*m))*np.sum((hyp(theta0,theta1,X)-Y)**2)


# calculate Batch Grads
def calculate_grads_batch(X,Y,theta0,theta1,m):
    dtheta0 = (1/m)*np.sum(hyp(theta0,theta1,X)-Y)
    dtheta1 = float((1/m)*X.T.dot(hyp(theta0,theta1,X)-Y))
    
    return dtheta0,dtheta1

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

def create_batches(X,y,batch_size):
    for i in range(0,X.shape[0],batch_size):
        batch = slice(i,i+batch_size)
        yield X[batch],y[batch]
        


def fit_model(X,Y,learning_rate = 0.001,tolerance = 1e-3,batch_size = 6,n_epochs = 20,plot_result=False,kind ="SGD"):
    theta0, theta1,m = intialize_params(X)

    # Batch Gradient descent
    if(kind=="batch"):
        iters = 0
        while(True):

            cost_old = compute_cost(theta0,theta1,X,Y,m)
            dtheta0 ,dtheta1 = calculate_grads_batch(X,Y,theta0,theta1,m)
            theta0,theta1 = update_params(theta0,theta1,learning_rate,dtheta0,dtheta1)
            cost_new = compute_cost(theta0,theta1,X,Y,m)
            iters += 1 
            
            if(np.linalg.norm(np.array([dtheta0,dtheta1]))<=tolerance):
                break
            elif(abs(cost_old-cost_new)<=tolerance):
                break
        
    # mini-batch Gradient descent
    if(kind == "mini-batch"):

        iters = 0


        while(True):

            
            for batch in create_batches(X,Y,batch_size):
                cost_old = compute_cost(theta0,theta1,batch[0],batch[1],batch_size)
                dtheta0 ,dtheta1 = calculate_grads_batch(batch[0],batch[1],theta0,theta1,batch_size)
                theta0,theta1 = update_params(theta0,theta1,learning_rate,dtheta0,dtheta1)
                cost_new = compute_cost(theta0,theta1,batch[0],batch[1],batch_size)
                
            
            iters += 1 
            if(np.linalg.norm(np.array([dtheta0,dtheta1]))<=tolerance):
                break
            elif(abs(cost_new-cost_old)<=tolerance):
                break
        
        

    # Stocastic Gradient descent
    if(kind == "SGD"):
        iters = 0
        while(True):
            
            
            for batch in create_batches(X,Y,1):
                cost_old = compute_cost(theta0,theta1,batch[0],batch[1],1)
                dtheta0 , dtheta1 = calculate_grads_stocastic(batch[0],batch[1],theta0,theta1)
                theta0, theta1 = update_params(theta0,theta1,learning_rate,dtheta0,dtheta1)
                cost_new = compute_cost(theta0,theta1,batch[0],batch[1],1)   
                
           
            iters +=1
            if(np.linalg.norm(np.array([dtheta0,dtheta1]))<=tolerance):
                break
            elif(abs(cost_new-cost_old)<=tolerance):
                break
    # plot fitted line w.r.t data

    if(plot_result):
        plt.scatter(x_points,y_points)

        points = np.linspace(0,10,10)
        predicted_line = theta0+theta1*points
        
        plt.plot(points,predicted_line,"orange")
        plt.grid()
        plt.legend(["Linear regression fit","Data"])

    
    return theta0,theta1,iters



theta0,theta1,iters = fit_model(x_points,y_points,kind="batch",plot_result=False)
print("batch : \ntheta0 = "+str(theta0)+"\ntheta1 = "+str(theta1)+"\nNumbers of itterations to converse : "+str(iters))
predictions = hyp(theta0,theta1,x_points)
print("R2 score :"+ str(r2_score(y_points,predictions))+"\n\n")

theta0,theta1,iters = fit_model(x_points,y_points,kind="SGD",plot_result=False,tolerance=1e-2)
print("SGD : \ntheta0 = "+str(theta0)+"\ntheta1 = "+str(theta1)+"\nNumbers of epochs to converse : "+str(iters))
predictions = hyp(theta0,theta1,x_points)
print("R2 score :"+ str(r2_score(y_points,predictions))+"\n\n")

theta0,theta1,iters = fit_model(x_points,y_points,kind="mini-batch",plot_result=True,batch_size=6)
print("mini-batch :\ntheta0 = "+str(theta0)+"\ntheta1 = "+str(theta1)+"\nNumbers of epochs to converse : "+str(iters))
predictions = hyp(theta0,theta1,x_points)
print("R2 score :"+ str(r2_score(y_points,predictions))+"\n\n")


plt.show()