import numpy as np
# Creating two random 3X3 matrices
A=np.random.randint(10,size=(3,3))
B=np.random.randint(10,size=(3,3))

#Basic operations
A+B             # Add
A*B             # Element Wise Multiplication (Hadamard Product)
A@B             # Matrix Product
A.T             # Transpose
np.linalg.det(A)# Determinant
np.linalg.inv(A)# Inverse

# Comparing product of matrix and its inverse with Identity Matrix
np.round(np.linalg.inv(A)@A) 
np.eye(3)

np.round(np.linalg.inv(A)@np.linalg.inv(B))==np.round(np.linalg.inv(B@A))
np.linalg.matrix_rank(A)
b=np.random.randint(10,size=(3,1))

import time
tic=time.time()
c=np.linalg.inv(A)@b
toc=time.time()
print(c)
print("Using Inverse: " + str((toc-tic)*1000) + "ms")
tic=time.time()
c=np.linalg.solve(A,b)
toc=time.time()
print(c)
print("Gaussian Elimination: " + str((toc-tic)*1000) + "ms")
#Eigenvalue Decomposition
eigenvalues, eigenvectors = np.linalg.eig(A)
eigenvalues*np.eye(3)
eigenvectors
np.trace(A) # Sum of all diagonal values
np.round(sum(eigenvalues))
np.round(A@eigenvectors)==np.round(eigenvectors@(eigenvalues*np.eye(3)))
np.round(A@eigenvectors)
np.round(eigenvectors@(eigenvalues*np.eye(3)))
A@A@A
eigenvectors@((eigenvalues**3)*np.eye(3))@np.linalg.inv(eigenvectors)
A
np.linalg.norm(A)
b
np.linalg.norm(b)

A=np.random.randint(10,size=(4,3))
A
np.linalg.inv(A)
np.linalg.pinv(A)
np.round(np.linalg.pinv(A)@A)
U,D,V = np.linalg.svd(A)
U
D
V
U@(D*np.eye(4))@V.T
np.round(U.T)==np.round(np.linalg.inv(U)) # U is Orthogonal
