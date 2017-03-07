# exercise 6.1.1

from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, show
from scipy.io import loadmat
from sklearn import cross_validation, tree
import numpy as np

# Load Matlab data file and extract variables of interest
mat_data = loadmat('../Data/wine2.mat')
X = mat_data['X']
y = mat_data['y'].squeeze()
attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
classNames = [name[0][0] for name in mat_data['classNames']]
N, M = X.shape
C = len(classNames)

# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 21, 1)

# Simple holdout-set crossvalidation
test_proportion = 0.5
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=test_proportion)

# Initialize variables
Error_train = np.empty((len(tc),1))
Error_test = np.empty((len(tc),1))

for i, t in enumerate(tc):
    # Fit decision tree classifier, Gini split criterion, different pruning levels
    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
    dtc = dtc.fit(X_train,y_train)

    # Evaluate classifier's misclassification rate over train/test data
    y_est_test = dtc.predict(X_test)
    y_est_train = dtc.predict(X_train)
    misclass_rate_test = sum(np.abs(y_est_test - y_test)) / float(len(y_est_test))
    misclass_rate_train = sum(np.abs(y_est_train - y_train)) / float(len(y_est_train))
    Error_test[i], Error_train[i] = misclass_rate_test, misclass_rate_train
    
f = figure(); f.hold(True)
plot(tc, Error_train)
plot(tc, Error_test)
xlabel('Model complexity (max tree depth)')
ylabel('Error (misclassification rate)')
legend(['Error_train','Error_test'])
    
show()    
