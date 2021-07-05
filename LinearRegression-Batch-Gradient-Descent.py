# import libraries
import numpy as np
import matplotlib.pyplot as plt

# training data
x_points =np.array( [[1,1,2,3,4,5,6,7,8,9,10,11]]).T
y_points = np.array([[1,2,3,1,4,5,6,4,7,10,15,9]]).T

# intialize parameters for the model
# also get the number of training points
def intialize_params(X):
    theta0 = 0
    theta1 = 0
    m = X.shape[1]
    return theta0 ,theta1,m

# function to calculate hypothisis h(X) 
def hyp(theta0,theta1,X):
    return theta0 + theta1*X


# function to compute cost
def compute_cost(theta0,theta1,X,Y,m):
    return (1/(2*m))*np.sum(((hyp(theta0,theta1,X)-Y))**2)

# function to calculate grads
def calculate_grads(X,Y,theta0,theta1,m):
    dtheta0 = (1/m)*np.sum(hyp(theta0,theta1,X)-Y)
    dtheta1 = (1/m)*np.sum(hyp(theta0,theta1,X)-Y)

    return dtheta0,dtheta1

# update paramaters with gradients.
def update_params(theta0,theta1,learning_rate,dtheta0,dtheta1):
    theta0 = theta0 - learning_rate*dtheta0;
    theta1 = theta1 - learning_rate*dtheta1;
    
    return theta0, theta1

"""
intialize theta0, theta1
compute intial cost
    while((old_cost-new_cost)<some threshold)
        1 - compute cost 
        2 - get gradient
        3 - update params
    return params
"""
def fit_model(X,Y,learning_rate = 0.001,itters = 50,tolerance = 1e-5,
                plot_result = False,show_iters = False,with_iters=False,with_tolerance = False):

    # intialize parameters
    theta0 , theta1 ,m= intialize_params(X)
    
    # using iterations as stop condtion
    if(with_iters):
        for i in range(itters):
            dtheta0,dtheta1 = calculate_grads(X,Y,theta0,theta1,m)
            theta0,theta1 = update_params(theta0,theta1,learning_rate,dtheta0,dtheta1)
            cost = compute_cost(theta0,theta1,X,Y,m)

            if(show_iters):
                print("caluculated cost at iteration number "+str(i)+ " :"+str(cost))

    # using tolerance as stop condtion
    if(with_tolerance):
        i =0
        while(True):
            cost_old = compute_cost(theta0,theta1,X,Y,m)
            dtheta0,dtheta1 = calculate_grads(X,Y,theta0,theta1,m)
            theta0,theta1 = update_params(theta0,theta1,learning_rate,dtheta0,dtheta1)
            cost_new = compute_cost(theta0,theta1,X,Y,m)
            i +=1
            if(show_iters):
                print("caluculated cost at iteration number "+str(i)+ " :"+str(cost_new))
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
    return theta0,theta1

def predict(x_point,theta0,theta1):
    prediction = hyp(theta0,theta1,x_point)
    print("prediction for point x = 8 is : "+str(prediction))
    return prediction

theta0,theta1 = fit_model(x_points,y_points,plot_result=True,show_iters=True,with_tolerance=True)
prediction = predict(theta0,theta1,8)



plt.show()