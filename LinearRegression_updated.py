"""
Generate a synthetic linear data with noise
Find the best linear model (optimum parameters) using normal equation (derived by equating derative to 0)
Find the same using gradient descent (iteratively)
Use standard library like scikit learn for the same problem
"""

import numpy as np
import matplotlib.pyplot as plt

# Generating Data
features, outputs, samples = 1,1,100  
X=np.random.rand(samples,features)                  # Input Design Matrix
Wrand=np.random.rand(features,outputs)*100          # Random Coefficents generated for data generation
Y= 4 + X@Wrand + np.random.randn(samples,outputs)   # Output Matrix

# Plotting Actual Data
plt.scatter(X,Y)

# Solving in one shot using normal equation
Wopt=(np.linalg.solve(X.T@X,X.T@Y))         # Using Normal Equation without bias
X_= np.hstack((np.ones((samples,1)),X))     #  
Wopt_=(np.linalg.solve(X_.T@X_,X_.T@Y))     # Using Normal Equation with bias
Hopt=X@Wopt                                 # Prediction without bias
Hopt_=X_@Wopt_                              # Prediction with bias

# Plotting Predicted and Actual Data
plt.scatter(X,Hopt)
plt.scatter(X,Y)
plt.scatter(X,Hopt_)

# Learning using gradient descent

W=np.zeros((features,outputs))              # Initializing weights for learning
b=np.zeros((outputs,))                      # Initializing bias for learning
alpha=0.01                                  # Learning rate as a hyperparameter

# Defining the mean square error cost function (regression task)
def costfunc(X,Y,W,b,samples):
    return np.sum((X@W + b - Y)**2)/samples

# Defining the derivative of cost function
def dcostfunc(X,Y,W,b,samples):
    return (X.T@(X@W+b-Y))*(2/samples) , (np.sum(X@W+b-Y))*(2/samples)

#Testing Cost Function 
costfunc(X,Y,Wopt,Wopt,samples)
costfunc(X,Y,W,b,samples)

epochs=10000                                  # Number of times to go through the whole data set (a hyperparameter)
costarr=np.zeros((epochs,))

# Using Gradient Descent for error minimisation
for i in range(epochs):
    costarr[i]=costfunc(X,Y,W,b,samples)      # Calculating cost for plotting
    dcostW, dcostb=dcostfunc(X,Y,W,b,samples) # Calculating gradient for learning
    W=W-alpha*dcostW                          # Weight Update
    b=b-alpha*dcostb                          # Bias Update
    
# Plotting error with respect to epochs    
plt.plot(np.arange(i+1),costarr)
    
# Final prediction after learning W and b
Hfinal= X@W+b

# Plot actual Data and predictions in one plot
plt.scatter(X,Y)
plt.scatter(X,Hfinal)

costfunc(X,Y,W,b,samples)    

# Using Standard Library for the same problem
from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(X,Y)              # Learning optimum pararmeters 
model.intercept_            # Bias b  - constant intercept terms for each output
model.coef_                 # Weight W - coefficients of features
model.score(X,Y)            # Final Mean Square Error after learning


# Defining cost function with regularisation 
# Defining the mean square error cost function (regression task) with regularisation (L2)
def costfunc_l2(X,Y,W,b,samples,lambda_):
    return np.sum((X@W + b - Y)**2)/samples + lambda_*np.linalg.norm(W)**2 
    # return (X@W + b - Y).T@(X@W + b - Y)/samples +lambda_*W.T@W

# Defining the derivative of cost function with regularisation (L2)
def dcostfunc_l2(X,Y,W,b,samples,lambda_):
    return (X.T@(X@W+b-Y))*(2/samples) + 2*lambda_*W , (np.sum(X@W+b-Y))*(2/samples)

#Using Stochastic Gradient Descent
def minimise(X,Y,W,b,lambda_,alpha):                              # samples can be replaced with Y.shape[0]            
    costarr[i]=costfunc_l2(X,Y,W,b,Y.shape[0],lambda_)            # Calculating cost for plotting
    dcostW, dcostb=dcostfunc_l2(X,Y,W,b,Y.shape[0],lambda_)       # Calculating gradient for learning
    W=W-alpha*dcostW                                              # Weight Update
    b=b-alpha*dcostb                                              # Bias Update
    return W,b

lambda_= 1
batch_size=10

batches = [] #Add code

for i in range(epochs):
    for batch in batches:
        W,b=minimise(batch_X,batch_Y,W,b)


#Hyperparameter optimisation
def batch(X,Y,batch_size):
    #Add code
    return batches

def optimise(X,Y,epochs,batch_size,alpha,lambda_):
    batches=batch(X,Y,batch_size)
    for i in range(epochs):
        for batch in batches:
            return minimise(batch_X,batch_Y,W,b,alpha)

hyperparameters={"alpha_list":[],"lambda_list":[],"batch_size_list":[]}

def grid search(X,Y,epochs,hyperparameters):
    cost_list=[]
    parameter_list=[]
    hyperparameter_list=[]
    for batch_size in hyperparameters["batch_size_list"]:
        for alpha in hyperparameters["alpha_list"]:
            for lambda_ in hyperparameters["lambda_list"] :   
                W,b=optimise(X,Y,epochs,batch_size,alpha,lambda_)
                cost = costfunc_l2(X,Y,W,b,Y.shape[0],lambda_)
                if cost < cost_list[-1]:
                    cost_list.append(cost)
                    hyperparameter_list.append([batch_size,alpha,lambda_])
                    parameter_list.append([W,b])

    return cost_list[-1], hyperparameter_list[-1], parameter_list[-1]


