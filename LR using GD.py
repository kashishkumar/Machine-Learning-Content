import numpy as np
import matplotlib.pyplot as plt
nx,ny,m= 1,1,100
X=np.random.rand(nx,m)
Wrand=np.random.rand(ny,nx)*100
Y= 4 + Wrand@X + np.random.randn(ny, m)
plt.scatter(X,Y)
W=np.zeros((ny,nx))
Wopt=(np.linalg.solve(X@X.T,X@Y.T)).T # Using Normal Equation without bias
X_= np.vstack((np.ones((1,m)),X))
Wopt_=(np.linalg.solve(X_@X_.T,X_@Y.T)).T # Using Normal Equation with bias
Hopt=Wopt@X
Hopt_=Wopt_@X_
plt.scatter(X,Y)
plt.scatter(X,Hopt)
plt.scatter(X,Hopt_)
b=np.zeros((ny,))
lr=0.01

def costfunc(X,Y,W,b,m):
    return np.sum((W@X + b - Y)**2)/m

def dcostfunc(X,Y,W,b,m):
    return ((W@X+b-Y)@X.T)*(2/m) , (np.sum(W@X+b-Y))*(2/m)

costfunc(X,Y,Wopt,Wopt,m)
costfunc(X,Y,W,b,m)

iterations=10000
costarr=np.zeros((iterations,))

# Using Gradient Descent for error minimisation
for i in range(iterations):
    costarr[i]=costfunc(X,Y,W,b,m)
    dcostW, dcostb=dcostfunc(X,Y,W,b,m)
    W=W-lr*dcostW
    b=b-lr*dcostb
    
    
plt.plot(np.arange(i+1),costarr)
    
Hfinal= W@X+b

plt.scatter(X,Y)
plt.scatter(X,Hfinal)

costfunc(X,Y,W,b,m)    

# Using Standard Library for the same problem
from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(X.T,Y.T)
model.intercept_
model.coef_
model.score(X.T,Y.T)


