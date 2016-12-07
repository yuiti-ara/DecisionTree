# external packages
import numpy as np
import pandas as pd


class Tree:
    def __init__(self, y, attr, cutoff):

        # target attribute distribution
        self.probs = y.sum(axis=0) / len(y)

        # decision attribute & decision value
        self.attr = attr if attr else "pure"
        self.cutoff = cutoff

        # child nodes
        self.left = None
        self.right = None

    def inference(self, x):

        # walk down the tree
        root = self
        while root:

            # return distribution if leaf node
            if not root.nodes:
                return root.probs

            # check if attribute value is less of equal than cutoff
            is_left = x[root.attr] <= root.cutoff

            # go to left child, else right child
            root = root.left if is_left else root.right


def height(root):

    if not root:
        return 0
    else:
        return max(height(root.left), height(root.right)) + 1


def entropy(y):

    if not y.size:
        return 0

    # compute value distribution
    probs = y.sum(axis=0) / y.shape[0]
    probs = probs[probs != 0]

    # compute entropy
    h = - sum(probs * np.log2(probs))

    return h


def info_gain(x, y, cutoff):

    # compute entropy before
    before = entropy(y)

    # divide into two subsets
    subset_l = y[x <= cutoff, :]
    subset_r = y[x > cutoff, :]

    # compute each weighted entropy
    h_l = len(subset_l) * entropy(subset_l)
    h_r = len(subset_r) * entropy(subset_r)

    # compute information gain
    ig = before - (h_l + h_r) / len(y)

    return ig


def select_attr(X, y):

    # no split if pure subset
    if len(y) in y.sum(axis=0):
        return None, None

    # list attributes to check
    attrs = list(range(X.shape[1]))

    # compute information-gain for each (attr, cutoff) combination
    gains = {}
    for attr in attrs:
        for cutoff in X[:, attr]:

            # compute information gain
            gain = info_gain(X[:, attr], y, cutoff)

            # store info-gain
            gains[(attr, cutoff)] = gain

    # pick attr & cutoff with highest info-gain
    attr, cutoff = max(gains, key=gains.get)

    return attr, cutoff


def splits(X, y, attr, limit):

    # return empty if no attribute
    if not attr:
        return None, None

    # define boolean idx vector
    lines = X[:, attr] <= limit

    # divide into two subsets
    subset_l = X[lines, :], y[lines]
    subset_r = X[~lines, :], y[~lines]

    return subset_l, subset_r


def grow(X, y):

    # pick attribute to split
    attr, cutoff = select_attr(X, y)

    # split data into subsets
    data_l, data_r = splits(X, y, attr, cutoff)

    # create tree-node
    root = Tree(y, attr, cutoff)

    # add child-nodes
    root.left = grow(*data_l) if data_l else None
    root.right = grow(*data_r) if data_r else None

    return root


def main():

    df = pd.read_csv("data/iris.csv", index_col=[0])
    X = df.drop("Species", axis=1).values
    y = pd.get_dummies(df["Species"]).values

    tree = grow(X, y)

    print(height(tree))

if __name__ == "__main__":
    main()