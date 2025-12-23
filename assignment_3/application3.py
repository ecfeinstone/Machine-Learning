import math
import numpy as np
from soft_svm import *
from sklearn.datasets import make_classification
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
Problem 2 (30 points in total): Apply your Soft SVM
In this problem, use your soft-margin SVM implemented in problem 1 to do binary classification.
Tune alpha, C, and n_epoch to achieve high test accuracy.
Report the three parameters you used (5 pts), along with train accuracy (5 pts), test accuracy (5 pts), and the objective value on the test set (5 pts).
You must have a traing and test accuracy greater than 90% (10 pts).
Note: Don't use any existing SVM package; use your own version.
'''
#--------------------------

n_samples = 200
X, y = make_classification(n_samples=n_samples, n_features=4, n_informative=4, n_redundant=0, n_clusters_per_class=1, random_state=1)
X = (X - X.mean(axis=0)) / X.std(axis=0)
y = 2*y - 1
Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]
C, alpha, n_epoch = 1.0, 0.1, 1000
w = train(Xtrain, Ytrain, C, alpha, n_epoch)
yhat_train = predict(Xtrain, w)
yhat_test = predict(Xtest, w)
train_accuracy = np.mean(yhat_train == Ytrain)
test_accuracy = np.mean(yhat_test == Ytest)
J_test = compute_J(Xtest, Ytest, w, C=1.0)
print("C:", C, "alpha:", alpha, "n_epoch:", n_epoch)
print("Train accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)
print("Test objective:", J_test)