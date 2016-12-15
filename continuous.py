import random

# external packages
import numpy as np
import pandas as pd


class Train:

    @staticmethod
    def accuracy(predicted, actual):
        return sum(Train.numeric(predicted) == Train.numeric(actual)) / len(actual)

    @staticmethod
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

    def predict(self, X):
        pred = np.zeros([len(X), len(self.probs)])
        for idx, x in enumerate(X):
            pred[idx, self._predict(x)] = 1
        return pred

    def _predict(self, x):

        # walk down the tree
        root = self
        while root:

            # return distribution if leaf node
            if not root.left and not root.right:

                # choose random when tie
                ps = np.where(root.probs == root.probs.max())
                p = np.random.choice(*ps)
                return p

            # check if attribute value is less or equal than cutoff
            is_left = x[root.attr] <= root.cutoff

            # go to left child, else right child
            root = root.left if is_left else root.right

    def height(self):
        if not self:
            return 0
        return max(Tree.height(self.left), Tree.height(self.right)) + 1


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
    def select_attr(cls, X, Y, max_n):

        # no split if pure subset
        if len(Y) in Y.sum(axis=0):
            return None, None

        # list attributes to check
        #attrs = list(range(X.shape[1]))
        attrs = np.random.choice(X.shape[1], max_n, replace=False)

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
            return None, None, None, None

        # define boolean idx vector
        lines = X[:, attr] <= cutoff

        # return two subsets
        return X[lines], Y[lines], X[~lines], Y[~lines]

    @classmethod
    def grow(cls, X, Y, max_n):

        # pick attribute to split
        attr, cutoff = cls.select_attr(X, Y, max_n)

        # split data into subsets
        X_l, Y_l, X_r, Y_r = cls.splits(X, Y, attr, cutoff)

        # create tree-node
        root = Tree(Y, attr, cutoff)

        # add child-nodes
        if attr:
            root.left = cls.grow(X_l, Y_l, max_n)
            root.right = cls.grow(X_r, Y_r, max_n)

        return root


def main():

    np.random.seed(0)

    df = pd.read_csv("data/iris.csv", index_col=[0])
    X = df.drop("Species", axis=1).values
    Y = pd.get_dummies(df["Species"]).values

    # grow tree
    tree = DecicionTree.grow(X, Y, max_n=1)
    y_pred = tree.predict(X)

    print(Train.accuracy(y_pred, Y))


if __name__ == "__main__":
    main()
