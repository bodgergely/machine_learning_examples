# https://deeplearningcourses.com/c/data-science-supervised-machine-learning-in-python
# https://www.udemy.com/data-science-supervised-machine-learning-in-python
# Decision Tree for continuous-vector input, binary output
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
from util import get_data, get_xor, get_donut
from datetime import datetime


def entropy(y):
    """
    maximum 1.0 inf gain happens on p1=0.5 and p0=0.5 - when values are half-half
    """
    # assume y is binary - 0 or 1
    N = len(y)
    s1 = (y == 1).sum()
    if 0 == s1 or N == s1:
        return 0
    p1 = float(s1) / N
    p0 = 1 - p1
    return -p0*np.log2(p0) - p1*np.log2(p1)


class TreeNode:
    def __init__(self, depth=0, max_depth=None):
        # print 'depth:', depth
        self.depth = depth
        self.max_depth = max_depth

    def fit(self, X, Y):
        if len(Y) == 1 or len(set(Y)) == 1:
            # base case, only 1 sample
            # another base case
            # this node only receives examples from 1 class
            # we can't make a split
            self.col = None
            self.split = None
            self.left = None
            self.right = None
            self.prediction = Y[0]
            print("All Y is the same, no reason splittin further: ", Y[0])

        else:
            D = X.shape[1]
            cols = range(D) 

            max_ig = 0
            best_col = None
            best_split = None
            for col in cols: # loop through the features
                ig, split = self.find_split(X, Y, col)  # find the information gain split on this feature
                # print "ig:", ig
                if ig > max_ig:
                    max_ig = ig
                    best_col = col
                    best_split = split

            if max_ig == 0:
                # nothing we can do
                # no further splits
                self.col = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = np.round(Y.mean())                # no ig achieved - no point in splitting simply take the mean of the Y
                print("IG is zero, prediction: ", self.prediction)
            else:
                self.col = best_col
                self.split = best_split

                if self.depth == self.max_depth:
                    print("Reached max depth: ", self.max_depth)
                    self.left = None
                    self.right = None
                    self.prediction = [
                        np.round(Y[X[:,best_col] < self.split].mean()),     # we reached max split depth - do prediction on split
                        np.round(Y[X[:,best_col] >= self.split].mean()),
                    ]
                else:
                    print("Splitting on col: ", self.col, " split: ", self.split)
                    # print "best split:", best_split
                    # split - create nodes for left and right tree  
                    left_idx = (X[:,best_col] < best_split)
                    # print "left_idx.shape:", left_idx.shape, "len(X):", len(X)
                    Xleft = X[left_idx]
                    Yleft = Y[left_idx]
                    self.left = TreeNode(self.depth + 1, self.max_depth)
                    self.left.fit(Xleft, Yleft)      # - left node(data/labels smaller than split value on selected feature)

                    right_idx = (X[:,best_col] >= best_split)
                    Xright = X[right_idx]
                    Yright = Y[right_idx]
                    self.right = TreeNode(self.depth + 1, self.max_depth)
                    self.right.fit(Xright, Yright)   # - right node(data/labels bigger than split value on selected feature)

    def find_split(self, X, Y, col):
        """
        find ig, split on given feature (col)
        """
        # print "finding split for col:", col
        x_values = X[:, col]    # consider only the current feature
        sort_idx = np.argsort(x_values)  # sorted indexes in of x values
        x_values = x_values[sort_idx]    # sorted x values
        y_values = Y[sort_idx]           # corresponding y values to the sorted x values

        # Note: optimal split is the midpoint between 2 points
        # Note: optimal split is only on the boundaries between 2 classes

        # if boundaries[i] is true
        # then y_values[i] != y_values[i+1]
        # nonzero() gives us indices where arg is true
        # but for some reason it returns a tuple of size 1
        boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0]  # indexes where the consecutive y values differ
        best_split = None
        max_ig = 0
        for b in boundaries:        # loop through these indexes where the consecutive y-s differ
            split = (x_values[b] + x_values[b+1]) / 2       # look at the average of the corresponding two x values
            ig = self.information_gain(x_values, y_values, split)  # see the information gain on this specific feature's possible split value
            if ig > max_ig:     # keep track of the max ig and which split achieved that
                max_ig = ig
                best_split = split
        return max_ig, best_split

    def information_gain(self, x, y, split):
        # assume classes are 0 and 1
        # print "split:", split
        y0 = y[x < split]       # y0 find the y-s where x is smaller than split      
        y1 = y[x >= split]      # y1 where x is bigger than the split
        N = len(y)
        y0len = len(y0)
        if y0len == 0 or y0len == N:
            return 0
        p0 = float(len(y0)) / N     # ratio of count smaller
        p1 = 1 - p0 #float(len(y1)) / N     # ratio bigger
        # print "entropy(y):", entropy(y)
        # print "p0:", p0
        # print "entropy(y0):", entropy(y0)
        # print "p1:", p1
        # print "entropy(y1):", entropy(y1)
        return entropy(y) - p0*entropy(y0) - p1*entropy(y1)     # does this split divide the data well? we want to divide ina  way such that the entropy is lower on the sides

    def predict_one(self, x):
        # use "is not None" because 0 means False
        if self.col is not None and self.split is not None:
            print("depth: {}, col: {}, split: {}".format( self.depth, self.col, self.split))
            feature = x[self.col]   # column on this tree node where we should split
            if feature < self.split:   # if feature val on given col is smaller than split value we go and search further in the left tree
                if self.left:
                    p = self.left.predict_one(x)
                else:
                    p = self.prediction[0]      # if there is no more nodes to go(reached max depth) we predict the left side
            else:
                if self.right:
                    p = self.right.predict_one(x)
                else:
                    p = self.prediction[1]      # if there is no more nodes to go(reached max depth) we predict the right side
        else:
            # corresponds to having only 1 prediction
            p = self.prediction             # there was no reason to split since we had just one label at this point
            print("Reached pred: {} at depth: {}".format(p, self.depth))

        return p

    def predict(self, X):
        N = len(X)
        P = np.zeros(N)
        for i in range(N):
            P[i] = self.predict_one(X[i])
        return P


# This class is kind of redundant
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, Y):
        self.root = TreeNode(max_depth=self.max_depth)
        self.root.fit(X, Y)

    def predict(self, X):
        return self.root.predict(X)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)
    def print(self):
        pass


if __name__ == '__main__':
    X, Y = get_data()
    print("num of examples: ", len(X))
    # try donut and xor
    # from sklearn.utils import shuffle
    # X, Y = get_xor()
    # # X, Y = get_donut()
    # X, Y = shuffle(X, Y)

    # only take 0s and 1s since we're doing binary classification
    idx = np.logical_or(Y == 0, Y == 1)
    X = X[idx]
    Y = Y[idx]

    # split the data
    Ntrain = len(Y) // 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    
    #model = DecisionTree()
    model = DecisionTree(max_depth=3)
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print("Training time:", (datetime.now() - t0))

    t0 = datetime.now()
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy:", (datetime.now() - t0))

    t0 = datetime.now()
    print("Test accuracy:", model.score(Xtest, Ytest))
    print("Time to compute test accuracy:", (datetime.now() - t0))
