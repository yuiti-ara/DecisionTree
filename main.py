# standard library
import math

# external packages
import networkx as nx
import pandas as pd
from PIL import Image


def draw(graphx):

    # write into pydot language, save to image file
    p = nx.drawing.nx_pydot.to_pydot(graphx)
    p.write_png("test.png")

    # open image file
    img = Image.open("test.png")
    img.show()


class DecicionTrees:

    @staticmethod
    def train(data):
        pass

        # insert tree-root on the queue

        # while still queue

            # pop a tree node

            # select best attribute to split - split_on(X)

            # add node childs to the tree - split(node, attribute)

            # push the child-nodes to the queue

        # return tree-root


def entropy(series):

    # compute unique values frequencies
    probs = series.value_counts() / len(series)

    # compute entropy
    h = - sum([p * math.log(p, 2) for p in probs])

    return h


def info_gain(df, target, attr):

    # compute entropy before
    before = entropy(df[target])

    # compute entropy given target value on each subset
    series = df[attr]
    entropies = []
    for value in series.unique():

        # compute subset given atrribute value
        subset = df[series == value]

        # compute attribute value weight
        weight = len(subset) / len(series)

        # compute weighted subset entropy on target values
        h = weight * entropy(subset[target])

        # store entropy
        entropies.append(h)

    # compute information gain
    ig = before - sum(entropies)

    return ig


def info_gain_ratio(df, target, attr):

    # compute information-gain & intrinsic value
    ig = info_gain(df, target, attr)
    iv = entropy(df[attr])

    # compute ratio
    ratio = ig / iv

    return ratio


def main():

    # load data & define target variable
    df = pd.read_csv("playtennis.csv")
    target = "play"
    print(df)

    # compute information gain on given attribute
    for attr in df:
        ratio = info_gain_ratio(df, target, attr)
        print(ratio)


if __name__ == "__main__":
    main()