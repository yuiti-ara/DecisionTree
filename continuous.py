# external packages
import numpy as np
import pandas as pd


def height(root):
    if not root:
        return 0
    else:
        return max(height(root.left), height(root.right)) + 1


def accuracy(predicted, actual):
    return (predicted == actual) / len(actual)


def numeric(Y):

    # create numeric vector
    m, n = Y.shape
    aux = np.zeros([m, 1])
    for idx in range(n):
        aux[Y[:, idx] == 1] = idx

    return aux


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

    def accuracy(self, X, Y):

        # count number of correct predictions
        hits = 0
        for (x, y) in zip(X, Y):

            # check most probable class
            p = self.inference(x)
            if y[p.argmax()] == 1:
                hits += 1

        # return frequency of correct predictions
        return hits / len(Y)

    def inference(self, x):

        # walk down the tree
        root = self
        while root:

            # return distribution if leaf node
            if not root.left and not root.right:
                return root.probs

            # check if attribute value is less or equal than cutoff
            is_left = x[root.attr] <= root.cutoff

            # go to left child, else right child
            root = root.left if is_left else root.right


class DecicionTree:

    @staticmethod
    def entropy(Y):

        if not Y.size:
            return 0

        # compute value distribution
        probs = Y.sum(axis=0) / Y.shape[0]
        probs = probs[probs != 0]

        # compute entropy
        h = - sum(probs * np.log2(probs))

        return h

    @classmethod
    def info_gain(cls, x, Y, cutoff):

        # compute entropy before
        before = cls.entropy(Y)

        # divide into two subsets
        subset_l = Y[x <= cutoff, :]
        subset_r = Y[x > cutoff, :]

        # compute each weighted entropy
        h_l = len(subset_l) * cls.entropy(subset_l)
        h_r = len(subset_r) * cls.entropy(subset_r)

        # compute information gain
        ig = before - (h_l + h_r) / len(Y)

        return ig

    @classmethod
    def select_attr(cls, X, Y):

        # no split if pure subset
        if len(Y) in Y.sum(axis=0):
            return None, None

        # list attributes to check
        attrs = list(range(X.shape[1]))

        # compute information-gain for each (attr, cutoff) combination
        gains = {}
        for attr in attrs:
            for cutoff in X[:, attr]:

                # compute information gain
                gain = cls.info_gain(X[:, attr], Y, cutoff)

                # store info-gain
                gains[(attr, cutoff)] = gain

        # pick attr & cutoff with highest info-gain
        attr, cutoff = max(gains, key=gains.get)

        # return best attribute to split
        if gains[(attr, cutoff)] > 0:
            return attr, cutoff
        return None, None

    @staticmethod
    def splits(X, Y, attr, cutoff):

        # return empty if no attribute
        if not attr:
            return None, None

        # define boolean idx vector
        lines = X[:, attr] <= cutoff

        # divide into two subsets
        subset_l = X[lines], Y[lines]
        subset_r = X[~lines], Y[~lines]

        return subset_l, subset_r

    @classmethod
    def grow(cls, X, Y):

        # pick attribute to split
        attr, cutoff = cls.select_attr(X, Y)

        # split data into subsets
        data_l, data_r = cls.splits(X, Y, attr, cutoff)

        # create tree-node
        root = Tree(Y, attr, cutoff)

        # add child-nodes
        root.left = cls.grow(*data_l) if data_l else None
        root.right = cls.grow(*data_r) if data_r else None

        return root


def main():

    df = pd.read_csv("data/iris.csv", index_col=[0])
    X = df.drop("Species", axis=1).values
    Y = pd.get_dummies(df["Species"]).values

    # grow tree
    tree = DecicionTree.grow(X, Y)

    # # inference
    # for data in zip(X, y):
    #     print(tree.inference(data[0]), data[1])

    print(tree.accuracy(X, Y))


if __name__ == "__main__":
    main()
