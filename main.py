# standard library
from collections import deque
import math

# external packages
import pandas as pd


class Tree:
    def __init__(self, df):

        # dataset
        self.df = df

        # target variable name & distribution
        self.target = None
        self.probs = {}

        # decision attribute name & child-nodes for its values
        self.attr = None
        self.nodes = {}

    def update(self, attr, subsets, target):

        # compute target variable distribution
        series = self.df[target]
        probs = series.value_counts() / len(series)

        # create child-nodes
        nodes = {value: Tree(subset) for value, subset in subsets.items()}

        # update node
        self.df = None
        self.target = target
        self.probs = probs.to_dict()
        self.attr = attr if attr else "pure"
        self.nodes = nodes

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
        probs = series.value_counts() / len(series)

        # compute entropy
        h = - sum([p * math.log(p, 2) for p in probs])

        return h

    @classmethod
    def info_gain(cls, df, target, attr):

        # compute entropy before
        before = cls.entropy(df[target])

        # pick subset for each attribute value
        values = df[attr]
        subsets = (df[values == value] for value in values.unique())

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
        values = df[attr]
        for value in values.unique():

            # pick subset
            subset = df[values == value]

            # drop attribute
            subset = subset.drop(attr, axis=1)

            # store subset
            subsets[value] = subset

        return subsets

    @classmethod
    def grow(cls, root, target):

        # pick attribute to split
        attr = cls.split_on(root.df, target)

        # split data into subsets
        subsets = cls.splits(root.df, attr)

        # update current node
        root.update(attr, subsets, target)

        # grow each child-node
        for node in root.nodes.values():
            cls.grow(node, target)

    @classmethod
    def get(cls, df, target):

        # define tree root
        tree = Tree(df)

        # grow tree from root
        cls.grow(tree, target)

        return tree


def main():

    # load data & define target variable
    df = pd.read_csv("playtennis.csv")
    target = "play"

    tree = DecicionTree.get(df, target)

    tree.print_bfs()

    test = {"outlook": "sunny",
            "temp": "hot",
            "humidity": "normal",
            "windy": False}
    result = tree.inference(test)
    print(result)


if __name__ == "__main__":
    main()
