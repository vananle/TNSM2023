import os
from math import sqrt

import numpy as np
from scipy.io import loadmat


def load_raw(args):
    # load ground truth
    data_path = os.path.join(args.datapath, 'dataset/{}.mat'.format(args.dataset))
    X = loadmat(data_path)['X']
    return X


def remove_outliers(data):
    q25, q75 = np.percentile(data, 25, axis=0), np.percentile(data, 75, axis=0)
    iqr = q75 - q25
    cut_off = iqr * 3
    lower, upper = q25 - cut_off, q75 + cut_off
    for i in range(data.shape[1]):
        flow = data[:, i]
        flow[flow > upper[i]] = upper[i]
        # flow[flow < lower[i]] = lower[i]
        data[:, i] = flow

    return data


def train_test_split(X, dataset):
    if 'abilene' in dataset:
        train_size = 4 * 7 * 288  # 4 weeks
        val_size = 288 * 7  # 1 week
        test_size = 288 * 7 * 2  # 2 weeks
    elif 'geant' in dataset:
        train_size = 96 * 7 * 4 * 2  # 2 months
        val_size = 96 * 7 * 2  # 2 weeks
        test_size = 96 * 7 * 4  # 1 month
    elif 'germany' in dataset:
        train_size = 96 * 7 * 4 * 2  # 2 months
        val_size = 96 * 7 * 2  # 2 weeks
        test_size = 96 * 7 * 4  # 1 month
    elif 'brain' in dataset:
        train_size = 1440 * 3  # 3 days
        val_size = 1440  # 1 day
        test_size = 1440 * 2  # 2 days
    elif 'uninett' in dataset:  # granularity: 1 hour
        train_size = 4 * 7 * 288  # 4 weeks
        val_size = 288 * 7  # 1 week
        test_size = 288 * 7 * 2  # 2 weeks
    elif 'renater_tm' in dataset:  # granularity: 5 min
        train_size = 4 * 7 * 288  # 4 weeks
        val_size = 288 * 7  # 1 week
        test_size = 288 * 7 * 2  # 2 weeks
    else:
        raise NotImplementedError

    X = np.reshape(X, newshape=[X.shape[0], -1])

    X_train = X[:train_size]

    X_val = X[train_size:val_size + train_size]

    X_test = X[val_size + train_size: val_size + train_size + test_size]

    if 'abilene' in dataset or 'geant' in dataset or 'brain' in dataset:
        X_train = remove_outliers(X_train)
        X_val = remove_outliers(X_val)

    return X_train, X_val, X_test


def prepare_tm_cycle(data, routing_cycle):
    x = []
    for t in range(routing_cycle, data.shape[0], routing_cycle):
        tm_cycle = data[t - routing_cycle:t + routing_cycle]
        if tm_cycle.shape[0] == routing_cycle + routing_cycle:
            x.append(tm_cycle)

    x = np.stack(x, axis=0)
    return x


def load_data(args):
    # loading dataset
    X = load_raw(args)
    train, val, test = train_test_split(X, args.dataset)

    n_node = X.shape[1]

    if len(train.shape) == 2:
        n_node = int(sqrt(train.shape[1]))
        train = np.reshape(train, newshape=(train.shape[0], n_node, n_node))
        val = np.reshape(val, newshape=(val.shape[0], n_node, n_node))
        test = np.reshape(test, newshape=(test.shape[0], n_node, n_node))

    train = prepare_tm_cycle(train, args.seq_len_x)
    val = prepare_tm_cycle(val, args.seq_len_x)
    test = prepare_tm_cycle(test, args.seq_len_x)

    return train, val, test, n_node
