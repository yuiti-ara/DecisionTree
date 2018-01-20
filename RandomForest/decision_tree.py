import numpy as np
import pandas as pd


np.random.seed(0)


def accuracy(predicted, actual):
    return sum(numeric(predicted) == numeric(actual)) / len(actual)


def numeric(Y):

    # create numeric vector
    m, n = Y.shape
    aux = np.zeros([m, 1])
    for idx in range(n):
        aux[Y[:, idx] == 1] = idx

    return aux


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


class DecisionTree:

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

    def info_gain(self, x, Y, cutoff):

        # compute entropy before
        before = self.entropy(Y)

        # divide into two subsets
        subset_l = Y[x <= cutoff, :]
        subset_r = Y[x > cutoff, :]

        # compute each weighted entropy
        h_l = len(subset_l) * self.entropy(subset_l)
        h_r = len(subset_r) * self.entropy(subset_r)

        # compute information gain
        ig = before - (h_l + h_r) / len(Y)

        return ig

    def _select_attr(self, X, Y, attrs):

        # compute information-gain for each (attr, cutoff) combination
        gains = {}
        for attr in attrs:
            for cutoff in X[:, attr]:

                # compute information gain
                gain = self.info_gain(X[:, attr], Y, cutoff)

                # store info-gain
                gains[(attr, cutoff)] = gain

        # pick attr & cutoff with highest info-gain
        attr, cutoff = max(gains, key=gains.get)

        # return best attribute to split
        return attr, cutoff, gains[(attr, cutoff)]

    def select_attr(self, X, Y, n_attrs):

        # check for single instance
        if len(X) <= 1:
            return None, None

        # check for pure subset
        if len(Y) in Y.sum(axis=0):
            return None, None

        # pick attributes to check
        attrs = np.random.choice(X.shape[1], n_attrs, replace=False)

        # pick best attribute to split
        attr, cutoff, gain = self._select_attr(X, Y, attrs)

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

    def grow(self, X, Y, n_attrs):

        # pick attribute to split
        attr, cutoff = self.select_attr(X, Y, n_attrs)

        # split data into subsets
        X_l, Y_l, X_r, Y_r = self.splits(X, Y, attr, cutoff)

        # grow root node
        node = Tree(Y, attr, cutoff)

        # grow child-nodes
        if attr:
            node.left = self.grow(X_l, Y_l, n_attrs)
            node.right = self.grow(X_r, Y_r, n_attrs)

        return node


if __name__ == '__main__':

    # load data
    df = pd.read_csv('/home/yuiti/PycharmProjects/RandomForest/data/iris.csv', index_col=[0])
    cols = ['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']
    df_X, df_Y = df[cols], pd.get_dummies(df['Species'])
    X, Y = df_X.values, df_Y.values

    # split data
    X_tr, Y_tr, X_ts, Y_ts = split_train_test(X, Y, test_size=.25)

    # grow tree
    model = DecisionTree()
    tree = model.grow(X_tr, Y_tr, n_attrs=4)
    y_pred = tree.predict(X_ts)

    acc = accuracy(y_pred, Y_ts)
    print(acc)
    assert float(acc) == 0.8918918918918919
