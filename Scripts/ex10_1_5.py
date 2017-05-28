# exercise 10_1_5
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import k_means


# Load Matlab data file and extract variables of interest
mat_data = loadmat('../Data/wildfaces.mat')
X = mat_data['X']
N, M = X.shape
# Image resolution and number of colors
x = 40 
y = 40
c = 3


# Number of clusters:
K = 10

# Number of repetitions with different initial centroid seeds
S = 1

# Run k-means clustering:
centroids, cls, inertia = k_means(X, K, verbose=True, max_iter=100, n_init=S)


# Plot results:

# Plot centroids
plt.figure(1)
n1 = np.ceil(np.sqrt(K/2)); n2 = np.ceil(np.float(K)/n1);
for k in range(K):
    plt.subplot(n1,n2,k+1)
    plt.imshow(np.reshape(centroids[k,:],(c,x,y)).T,interpolation='None',cmap=plt.cm.binary)
    plt.xticks([]); plt.yticks([])
    if k==np.floor((n2-1)/2): plt.title('Centroids')

# Plot few randomly selected faces and their nearest centroids    
L = 5       # number of images to plot
j = np.random.randint(0, N, L)
plt.figure(2)
for l in range(L):
    plt.subplot(2,L,l+1)
    plt.imshow(np.resize(X[j[l],:],(c,x,y)).T,interpolation='None',cmap=plt.cm.binary)
    plt.xticks([]); plt.yticks([])
    if l==np.floor((L-1)/2): plt.title('Randomly selected faces and their centroids')
    plt.subplot(2,L,L+l+1)
    plt.imshow(np.resize(centroids[cls[j[l]],:],(c,x,y)).T,interpolation='None',cmap=plt.cm.binary)
    plt.xticks([]); plt.yticks([])

plt.show()