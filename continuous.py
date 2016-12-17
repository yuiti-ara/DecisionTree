# external packages
import numpy as np
import pandas as pd


class Metric:

    @staticmethod
    def accuracy(predicted, actual):
        return sum(Metric.numeric(predicted) == Metric.numeric(actual)) / len(actual)

    @staticmethod
    def numeric(Y):

        # create numeric vector
        m, n = Y.shape
        aux = np.zeros([m, 1])
        for idx in range(n):
            aux[Y[:, idx] == 1] = idx

        return aux

    @staticmethod
    def split_train_test(X, Y, test_size=.2):
        ts_lines = np.random.choice(len(X), int(test_size*len(X)), replace=False)
        tr_lines = [line for line in np.arange(len(X)) if line not in ts_lines]
        return X[tr_lines, :], Y[tr_lines, :], X[ts_lines, :], Y[ts_lines, :]


class Tree:
    def __init__(self, y, attr, cutoff):

        # class distribution
        self.probs = y.sum(axis=0) / len(y)

        # decision attribute & decision value
        self.attr = attr
        self.cutoff = cutoff

        # child nodes
        self.left = None
        self.right = None

    def predict(self, X, dist=False):

        # create prediction matrix
        shape = [len(X), len(self.probs)]
        pred = np.zeros(shape)

        # fill prediction matrix
        for idx, x in enumerate(X):

            # get probabilities
            probs = self._predict(x)

            # set distrubution
            pred[idx, :] = self.hot_encode(probs) if not dist else probs

        return pred

    def _predict(self, x):

        # walk down the tree
        root = self
        while root:

            # return distribution if leaf node
            if not root.attr:
                return root.probs

            # check if attribute value is less or equal than cutoff
            is_left = x[root.attr] <= root.cutoff

            # go to left child, else right child
            root = root.left if is_left else root.right

    @staticmethod
    def hot_encode(probs):

        # pick all idx with highest value
        indices = np.where(probs == probs.max())

        # choose one
        idx = np.random.choice(*indices)

        # create hot-encoded vector
        hot = np.zeros(probs.shape)
        hot[idx] = 1

        return hot

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
    def _select_attr(cls, X, Y, attrs):

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
        return attr, cutoff, gains[(attr, cutoff)]

    @classmethod
    def select_attr(cls, X, Y, n_attrs):

        # check for single instance
        if len(X) <= 1:
            return None, None

        # check for pure subset
        if len(Y) in Y.sum(axis=0):
            return None, None

        # pick attributes to check
        attrs = np.random.choice(X.shape[1], n_attrs, replace=False)

        # pick best attribute to split
        attr, cutoff, gain = cls._select_attr(X, Y, attrs)

        # check for relevant gain
        if np.around(gain, 6) <= 0:
            return None, None

        # return best split
        return attr, cutoff

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
    def grow(cls, X, Y, n_attrs):

        # pick attribute to split
        attr, cutoff = cls.select_attr(X, Y, n_attrs)

        # split data into subsets
        X_l, Y_l, X_r, Y_r = cls.splits(X, Y, attr, cutoff)

        # grow root node
        root = Tree(Y, attr, cutoff)

        # grow child-nodes
        if attr:
            root.left = cls.grow(X_l, Y_l, n_attrs)
            root.right = cls.grow(X_r, Y_r, n_attrs)

        return root


class Forest:

    def __init__(self, trees):
        self.trees = trees
        self.classes = len(trees[0].probs)

    def predict(self, X, dist=False):

        # get avg of all trees
        pred = np.zeros([len(X), self.classes])
        for tree in self.trees:
            pred += tree.predict(X, dist=True)
        pred /= len(self.trees)

        # return ditribution
        if dist:
            return pred

        # return hot encoded
        for idx, probs in enumerate(pred):

            # choose most probable column
            p_indices = np.where(probs == probs.max())
            p_idx = np.random.choice(*p_indices)
            pred[idx, :] = 0
            pred[idx, p_idx] = 1

        return pred


class RandomForest:

    @staticmethod
    def bootstrap(X, Y):
        # random pick line indices with replacement
        lines = np.random.choice(len(X), len(X), replace=True)
        return X[lines, :], Y[lines, :]

    @classmethod
    def grow(cls, X, Y, n_trees=100, n_attrs=None):

        # default number of attributes
        if not n_attrs:
            n_attrs = int(np.sqrt(X.shape[1]))

        trees = []
        for _ in range(n_trees):

            # bootstrap sample
            X_boot, Y_boot = cls.bootstrap(X, Y)

            # grow tree
            tree = DecicionTree.grow(X_boot, Y_boot, n_attrs)

            # store tree
            trees.append(tree)

        return Forest(trees)


def main():

    # seed rng
    np.random.seed(0)

    # load data
    df = pd.read_csv("data/iris.csv", index_col=[0])
    X = df.drop("Species", axis=1).values
    Y = pd.get_dummies(df["Species"]).values

    # split data
    X_tr, Y_tr, X_ts, Y_ts = Metric.split_train_test(X, Y, test_size=.25)

    # grow tree
    tree = DecicionTree.grow(X_tr, Y_tr, n_attrs=4)
    y_pred = tree.predict(X_ts)
    print(Metric.accuracy(y_pred, Y_ts))

    # grow forest
    forest = RandomForest.grow(X_tr, Y_tr)
    y_pred = forest.predict(X_ts)
    print(Metric.accuracy(y_pred, Y_ts))

if __name__ == "__main__":
    main()
