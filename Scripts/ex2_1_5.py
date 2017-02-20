# exercise 2.2.4
from ex2_1_1 import *
from scipy.linalg import svd

# (requires data structures from ex. 2.2.1 and 2.2.3)
Y = X - np.ones((N,1))*X.mean(0)
U,S,V = svd(Y,full_matrices=False)
V=V.T


print(V[:,1].T)
## Projection of water class onto the 2nd principal component.
# Note Y is a numpy matrix, while V is a numpy array. 

# Either convert V to a numpy.mat and use * (matrix multiplication)
print((Y[y.A.ravel()==4,:] * np.mat(V[:,1]).T).T)

# Or interpret Y as a numpy.array and use @ (matrix multiplication for np.array)
#print( (np.asarray(Y[y.A.ravel()==4,:]) @ V[:,1]).T )
