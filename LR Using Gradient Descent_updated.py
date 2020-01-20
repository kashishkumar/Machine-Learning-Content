import numpy as np
import matplotlib.pyplot as plt

#Generating Data
features, outputs, samples = 1,1,100  
X=np.random.rand(features,samples)
Wrand=np.random.rand(outputs,features)*100
Y= 4 + Wrand@X + np.random.randn(outputs,samples)

#Plotting Actual Data
plt.scatter(X,Y)
Wopt=(np.linalg.solve(X@X.T,X@Y.T)).T       # Using Normal Equation without bias
X_= np.vstack((np.ones((1,samples)),X))
Wopt_=(np.linalg.solve(X_@X_.T,X_@Y.T)).T   # Using Normal Equation with bias
Hopt=Wopt@X                                 # Prediction without bias
Hopt_=Wopt_@X_                              # Prediction with bias
plt.scatter(X,Y)
plt.scatter(X,Hopt)
plt.scatter(X,Hopt_)
W=np.zeros((outputs,features))              # Initialing weights for learning
b=np.zeros((outputs,))                      #Initializing bias for learning
alpha=0.01                                  #Learning rate as a hyperparameter

# Defining the mean square error cost function
def costfunc(X,Y,W,b,m):
    return np.sum((W@X + b - Y)**2)/samples

#Defining the deriavtive of cost function
def dcostfunc(X,Y,W,b,m):
    return ((W@X+b-Y)@X.T)*(2/samples) , (np.sum(W@X+b-Y))*(2/samples)

#Testing Cost Function 
costfunc(X,Y,Wopt,Wopt,samples)
costfunc(X,Y,W,b,samples)

epochs=10000 # Number of times to go through the whole data set (a hyperparameter)
costarr=np.zeros((epochs,))

# Using Gradient Descent for error minimisation
for i in range(epochs):
    costarr[i]=costfunc(X,Y,W,b,samples)      # Calculating cost for plotting
    dcostW, dcostb=dcostfunc(X,Y,W,b,samples) # Calculating gradient for learning
    W=W-alpha*dcostW                          # Weight Update
    b=b-alpha*dcostb                          # Bias Update
    
# Plotting error with respect to epochs    
plt.plot(np.arange(i+1),costarr)
    
#Final prediction after learning W and b
Hfinal= W@X+b

#Plot actual Data and predictions in one plot
plt.scatter(X,Y)
plt.scatter(X,Hfinal)

costfunc(X,Y,W,b,samples)    

# Using Standard Library for the same problem
from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(X.T,Y.T)          #Learning optimum pararmeters 
model.intercept_            #Bias b  - constant intercept terms for each output
model.coef_                 #Weight W - coefficients of features
model.score(X.T,Y.T)        #Final Mean Square Error after learning