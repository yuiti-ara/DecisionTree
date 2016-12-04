# standard library
from collections import deque
import math

# external packages
import pandas as pd
import numpy as np


class Tree:
    def __init__(self, series, attr):

        # target attribute distribution & name
        probs = series.value_counts(normalize=True)
        self.probs = probs.to_dict()
        self.target = series.name

        # decision attribute name & empty child nodes
        self.attr = attr if attr else "pure"
        self.nodes = {}

    def inference(self, values):

        root = self
        while root:
            if not root.nodes:
                return root.probs
            root = root.nodes[values[root.attr]]

    def print_bfs(self):

        queue = deque([self])
        while queue:

            # pick node
            root = queue.popleft()
            print(root.attr, "\n", root.target, root.probs)

            # enqueue child-nodes
            for node in root.nodes.values():
                queue.append(node)


class DecicionTree:

    @staticmethod
    def entropy(series):

        # compute unique values probabilities
        probs = series.value_counts(normalize=True)

        # compute entropy
        h = - sum([p * math.log(p, 2) for p in probs])

        return h

    @classmethod
    def info_gain(cls, df, target, attr):

        # compute entropy before
        before = cls.entropy(df[target])

        # pick subset for each attribute value
        series = df[attr]
        subsets = (df[series == value] for value in series.unique())

        # compute each weighted subset entropy given target variable
        entropies = (len(subset) * cls.entropy(subset[target]) for subset in subsets)

        # compute information gain
        ig = before - sum(entropies) / len(df)

        return ig

    @classmethod
    def info_gain_ratio(cls, df, target, attr):

        # compute information-gain & intrinsic value
        ig = cls.info_gain(df, target, attr)
        iv = cls.entropy(df[attr])

        # compute ratio
        ratio = ig / iv

        return ratio

    @classmethod
    def split_on(cls, df, target):

        # no split if pure subset
        domain = df[target].unique()
        if len(domain) == 1:
            return ""

        # list attributes to check
        attrs = (attr for attr in df if attr != target)

        # find atrribute with hightest information-gain
        best = max(attrs, key=lambda attr: cls.info_gain_ratio(df, target, attr))

        return best

    @staticmethod
    def splits(df, attr):

        # return if no attribute
        if not attr:
            return {}

        # pick data subset with each attribute value
        subsets = {}
        series = df[attr]
        for value in series.unique():

            # pick subset
            subset = df[series == value]

            # drop attribute
            subset = subset.drop(attr, axis=1)

            # store subset
            subsets[value] = subset

        return subsets

    @classmethod
    def grow(cls, df, target):

        # pick attribute to split
        attr = cls.split_on(df, target)

        # split data into subsets
        subsets = cls.splits(df, attr)

        # create tree-node
        root = Tree(df[target], attr)

        # add child-nodes
        root.nodes = {value: cls.grow(subset, target) for value, subset in subsets.items()}

        return root


def entropy(y):

    if not y.size:
        return 0

    # compute value distribution
    probs = y.sum(axis=0) / len(y)
    probs = probs[probs != 0]

    # compute entropy
    h = - sum(probs * np.log2(probs))

    return h


def info_gain(x, y, limit):

    # compute entropy before
    before = entropy(y)

    # divide into two subsets
    subsets = y[x <= limit, :], y[x > limit, :]

    # compute each weighted entropy
    entropies = (len(subset) * entropy(subset) for subset in subsets)

    # compute information gain
    ig = before - sum(entropies) / len(y)

    return ig


def intrinsic_value(x, limit):

    # create two indicator columns
    indicators = np.column_stack([x <= limit, x > limit])

    # compute entropy
    iv = entropy(indicators)

    return iv


def info_gain_ratio(x, y, limit=None):

    if not limit:
        limit = x.mean()

    # compute information-gain & intrinsic value
    ig = info_gain(x, y, limit)
    iv = intrinsic_value(x, limit)

    # compute ratio
    ratio = ig / iv

    return ratio


def split_on(X, y):

    # no split if pure subset
    if y.ndim == 1:
        return 0

    # list column indices to check
    _, n = X.shape
    indices = list(range(n))

    # pick column idx with highest information gain
    best = max(indices, key=lambda idx: info_gain_ratio(X[:, idx], y))

    return best


def main():

    # # load data & define target variable
    # df = pd.read_csv("playtennis.csv")
    # target = "play"
    #
    # tree = DecicionTree.grow(df, target)
    # tree.print_bfs()

    # test = {"outlook": "sunny",
    #         "temp": "hot",
    #         "humidity": "normal",
    #         "windy": False}
    # result = tree.inference(test)
    # print(result)

    df = pd.read_csv("iris.csv", index_col=[0])
    X = df.drop("Species", axis=1).values
    y = pd.get_dummies(df["Species"]).values

    idx = split_on(X, y)
    print(idx)

if __name__ == "__main__":
    main()
