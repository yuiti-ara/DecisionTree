import numpy as np
import pandas as pd

from models import DecisionTree, RandomForest, DecicionTreeCategorical


np.random.seed(0)


def accuracy(predicted, actual):
    value = sum(numeric(predicted) == numeric(actual)) / len(actual)
    return float(value)


def numeric(Y):
    m, n = Y.shape
    aux = np.zeros([m, 1])
    for idx in range(n):
        aux[Y[:, idx] == 1] = idx
    return aux


def split_train_test(X, Y, test_size=.2):
    ts_lines = np.random.choice(len(X), int(test_size*len(X)), replace=False)
    tr_lines = [line for line in np.arange(len(X)) if line not in ts_lines]
    return X[tr_lines, :], Y[tr_lines, :], X[ts_lines, :], Y[ts_lines, :]


def load_data():
    df = pd.read_csv('/home/yuiti/PycharmProjects/RandomForest/data/iris.csv', index_col=[0])
    cols = ['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']
    df_X, df_Y = df[cols], pd.get_dummies(df['Species'])
    X, Y = df_X.values, df_Y.values
    return split_train_test(X, Y, test_size=.25)


def load_data_categorical():
    df = pd.read_csv('/home/yuiti/PycharmProjects/RandomForest/data/playtennis.csv')
    target = 'play'
    test = {
        'outlook': 'sunny',
        'temp': 'hot',
        'humidity': 'normal',
        'windy': False
    }
    return df, target, test


if __name__ == '__main__':

    X_tr, Y_tr, X_ts, Y_ts = load_data()

    # decision tree
    model = DecisionTree(n_attrs=4)
    model.fit(X_tr, Y_tr)
    y_pred = model.predict(X_ts)
    assert accuracy(y_pred, Y_ts) == 0.7837837837837838

    # random forest
    model = RandomForest()
    model.fit(X_tr, Y_tr)
    y_pred = model.predict(X_ts)
    assert accuracy(y_pred, Y_ts) == 1

    # decision tree categorical
    df, attr_targe, record_test = load_data_categorical()
    tree = DecicionTreeCategorical()
    tree.fit(df, attr_targe)
    print(tree.predict_one(record_test))
