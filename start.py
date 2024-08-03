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

#https://medium.com/geekculture/svm-classification-with-sklearn-svm-svc-how-to-plot-a-decision-boundary-with-margins-in-2d-space-7232cb3962c0


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
from itertools import combinations 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools

X1_df = pd.read_csv("./data/Data1Train.csv")
X1_df.head()

X1T_df = pd.read_csv("./data/Data1Test.csv")
X1T_df.head()

X1_train = X1_df[['x', 'y']]
Y1_train = X1_df[['Class']]

X1_test = X1T_df[['x', 'y']]
Y1_test = X1T_df[['Class']]


svc = SVC(kernel="linear",degree=1)
svc.fit(X1_train, Y1_train)
pred = svc.predict(X1_test)

print(list(X1_train['x'])[0:10])
print(list(Y1_train)[0:10])

plt.figure(figsize=(10, 8))
# Plotting our two-features-space
sns.scatterplot(x=list(X1_train['x'])[0:10], 
                y=list(Y1_train)[0:10], s=8)
# Constructing a hyperplane using a formula.
w = svc.coef_[0]           # w consists of 2 elements
b = svc.intercept_[0]      # b consists of 1 element
x_points = np.linspace(-1, 1)    # generating x-points from -1 to 1
y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points
# Plotting a red hyperplane
plt.plot(x_points, y_points, c='r')




















def generate_sublists(input_list): 
    sublists = [] 
    # Generate all sublists of length 2 
    sublists_of_2 = list(combinations(input_list, 2)) 
    sublists.extend(sublists_of_2) # Generate all sublists of length 3 
    sublists_of_3 = list(combinations(input_list, 3)) 
    sublists.extend(sublists_of_3) 
    return sublists



















