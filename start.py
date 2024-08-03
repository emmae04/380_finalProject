# using sklearn.svm to construct boundaries (to choose an optimtal boundary)
# calculate measure of purity (measure of how well stuff is clustered) sklearn metrics
# -> https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
# -> for each class
# S+ are samples at or above boundary (g)
# S- are samples below boundary  
# Gini index (measure of how impure a dataset is - 0 is a pure dataset)
# -> For every class, find a gini index 
# -> def gini(x):
    #return np.sum(np.abs(np.subtract.outer(x, x)))/(2*len(x)**2*x.mean())
# pure - all samples belong to the same class 


# first graph
# linear boundary
# distinguish 1/2 classes using a boundary 
# iterate thru each class combo to create boundary for each case


# X2_df = pd.read_csv("./data/Data2Train.csv")
# X3_df = pd.read_csv("./data/Data3Train.csv")
# X2_train = X2_df(['x', 'y'])
# Y2_train = X2_df(['Class'])

# X3_train = X3_df(['x', 'y'])
# Y3_train = X3_df(['Class'])


import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import itertools

X1_df = pd.read_csv("./data/Data1Train.csv")
X1_df.head()

X1T_df = pd.read_csv("./data/Data1Test.csv")
X1T_df.head()

X1_train = X1_df[['x', 'y']]
Y1_train = X1_df[['Class']]

X1_test = X1T_df[['x', 'y']]
Y1_test = X1T_df[['Class']]



X1, y1 = make_classification(n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)
clf = SVC(kernel="linear",degree=1)
clf.fit(X1_train, Y1_train)
pred = clf.predict(X1_test)

print(accuracy_score(Y1_test, pred))
print(confusion_matrix(Y1_test, pred))
print(clf.coef_)
















