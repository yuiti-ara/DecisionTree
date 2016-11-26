# standard library
from collections import deque
import math

# external packages
import pandas as pd


class DecicionTree:
    pass

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


def split_on(df, target):

    best_attr = ""
    best_ig = 0
    for attr in df:

        # skip target column
        if attr == target:
            continue

        # check gain-ratio
        ig = info_gain_ratio(df, target, attr)
        if ig > best_ig:
            best_attr, best_ig = attr, ig

    return best_attr


def splits(df, attr, target):

    # pick atrribute values
    values = df[attr].unique()

    # for each value, build leaf
    sub_tree = {}
    for value in values:

        # pick subset
        subset = df[df[attr] == value]

        # drop attribute
        subset = subset.drop(attr, axis=1)

        # check if pure
        domain = subset[target].unique()
        is_pure = len(domain) == 1

        # grow tree
        sub_tree[value] = domain[0] if is_pure else subset

    return {attr: sub_tree}


def train(df, target):

    tree = {"root": {}}
    queue = deque([(tree, "root", df)])
    while queue:

        # pick leaf
        subtree, label, df = queue.popleft()

        if type(df) != pd.DataFrame:
            continue

        # pick attribute to split
        attr = split_on(df, target)

        # grow tree
        subtree[label] = splits(df, attr, target)

        for value, subset in subtree[label][attr].items():
            queue.append((subtree[label][attr], value, subset))

    return tree["root"]


def main():

    # load data & define target variable
    df = pd.read_csv("playtennis.csv")
    target = "play"

    tree = train(df, target)

    from pprint import pprint
    pprint(tree)


if __name__ == "__main__":
    main()
