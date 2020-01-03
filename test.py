import numpy as np
a=np.array([1,3,5])
b=np.array([[2,4,5],[4,8,11],[2,5,6]])
b.mean()
np.corrcoef(b)
np.linalg.det(b)
np.random.randint(1,10,size=(3,3))
np.linalg.solve(b,a)
