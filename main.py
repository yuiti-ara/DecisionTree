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


def entropy(df, target):

    # compute target value frequencies
    probs = df[target].value_counts() / len(df)

    # compute entropy
    h = - sum([p * math.log(p, 2) for p in probs])

    return h


def information_gain(df, target, attr):

    # compute entropy before
    H = entropy(df, target)

    # compute sum of entropy on each attribute value
    entropies = []
    for value in df[attr].unique():

        # compute subset given atrribute value
        cond = df[attr] == value
        subset = df[cond]

        # compute attribute value weight
        weight = len(subset) / len(df)

        # compute weighted subset entropy
        h = weight * entropy(subset, target)

        # store entropy
        entropies.append(h)

    # compute information gain
    ig = H - sum(entropies)

    return ig


def intrinsic_value(df, attr):

    # compute intrinsic value
    ivs = []
    for value in df[attr].unique():

        # compute subset given atrribute value
        cond = df[attr] == value
        subset = df[cond]

        # compute subset weight
        weight = len(subset) / len(df)

        # add to the intrinsic value
        aux = weight * math.log(weight, 2)

        # store
        ivs.append(aux)

    # compute intrinsic value
    iv = - sum(ivs)

    return iv


def gain_ratio(df, target, attr):

    # compute information-gain & intrinsic value
    ig = information_gain(df, target, attr)
    iv = intrinsic_value(df, attr)

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
        ratio = gain_ratio(df, target, attr)
        print(ratio)


if __name__ == "__main__":
    main()