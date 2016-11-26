# standard library
from collections import deque
import math

# external packages
import pandas as pd


class Tree:
    def __init__(self, df):
        self.df = df
        self.attr = None
        self.nodes = {}


class DecicionTree:

    @staticmethod
    def entropy(series):

        # compute unique values frequencies
        probs = series.value_counts() / len(series)

        # compute entropy
        h = - sum([p * math.log(p, 2) for p in probs])

        return h

    @classmethod
    def info_gain(cls, df, target, attr):

        # compute entropy before
        before = cls.entropy(df[target])

        # compute entropy given target value on each subset
        series = df[attr]
        entropies = []
        for value in series.unique():

            # compute subset given atrribute value
            subset = df[series == value]

            # compute attribute value weight
            weight = len(subset) / len(series)

            # compute weighted subset entropy on target values
            h = weight * cls.entropy(subset[target])

            # store entropy
            entropies.append(h)

        # compute information gain
        ig = before - sum(entropies)

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

        best_attr = ""
        best_ig = 0
        for attr in df:

            # skip target column
            if attr == target:
                continue

            # check gain-ratio
            ig = cls.info_gain_ratio(df, target, attr)
            if ig > best_ig:
                best_attr, best_ig = attr, ig

        return best_attr

    @staticmethod
    def splits(df, attr, target):

        # pick atrribute values
        values = df[attr].unique()

        # for each value, build leaf
        nodes = {}
        for value in values:

            # pick subset
            subset = df[df[attr] == value]

            # drop attribute
            subset = subset.drop(attr, axis=1)

            # check if pure
            domain = subset[target].unique()
            is_pure = len(domain) == 1

            # grow tree
            nodes[value] = domain[0] if is_pure else Tree(subset)

        return nodes

    @classmethod
    def train(cls, df, target):

        tree = Tree(df)
        queue = deque([tree])
        while queue:

            # pick node
            root = queue.popleft()

            # pick attribute to split
            attr = cls.split_on(root.df, target)

            # split into child-nodes
            nodes = cls.splits(root.df, attr, target)

            # update node
            root.attr = attr
            root.nodes = nodes

            # enqueue leafs
            for node in nodes.values():

                # skip pure leafs
                if type(node) != Tree:
                    continue
                queue.append(node)

        return tree

    @staticmethod
    def print(tree):

        queue = deque([tree])
        while queue:

            # pick node
            root = queue.popleft()
            if type(root) == str:
                print(root)
            else:
                print(root.attr, root.nodes.keys())

            # enqueue leafs
            if type(root) == str:
                continue
            for node in root.nodes.values():
                # skip pure leafs
                queue.append(node)
            print("new lvl\n")

    @staticmethod
    def inference(root, values):

        while root:
            if type(root) == str:
                return root
            root = root.nodes[values[root.attr]]


def main():

    # load data & define target variable
    df = pd.read_csv("playtennis.csv")
    target = "play"

    tree = DecicionTree.train(df, target)
    #DecicionTree.print(tree)

    test = {"outlook": "sunny",
            "temp": "hot",
            "humidity": "normal",
            "windy": False}

    result = DecicionTree.inference(tree, test)
    print(result)


if __name__ == "__main__":
    main()
