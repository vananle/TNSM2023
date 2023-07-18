import operator
import os.path
import pickle
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
import torch
from scipy.io import loadmat
from scipy.sparse import linalg

DEFAULT_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load dataset ', pickle_file, ':', e)
        raise
    return pickle_data


ADJ_CHOICES = ['scalap', 'normlap', 'symnadj', 'transition', 'identity']


def load_adj(filename, adjtype):
    adj_mx = loadmat(filename)['A'].astype(np.float)

    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        raise NotImplementedError("adj type not defined")
    return adj


def make_graph_inputs(args, device):
    aptinit = None
    if not args.aptonly:
        adj_filename = os.path.join(args.adjdata_path, 'adjmx_{}.mat'.format(args.dataset))
        adj_mx = load_adj(adj_filename, args.adjtype)
        supports = [torch.tensor(i).to(device) for i in adj_mx]
        aptinit = None if args.randomadj else supports[0]  # ignored without do_graph_conv and add_apt_adj
    else:
        if not args.addaptadj and args.do_graph_conv: raise ValueError(
            'WARNING: not using adjacency matrix')
        supports = None
    return aptinit, supports


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    '''
    Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Arguments:
        patience: Number of epochs to wait after min has been hit.
                  After this number of no improvement, training stops.
    '''

    def __init__(self, patience=30):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("\n Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


def TRE(truth_seq, est_seq):
    # truth_seq and est_seq are test_sizex12x12 arrays
    res = [0] * truth_seq.shape[0]
    for i in range(truth_seq.shape[0]):
        res[i] = np.linalg.norm(est_seq[i] - truth_seq[i]) / np.linalg.norm(truth_seq[i])
    return res


def SRE(truth_seq, est_seq):
    # truth_seq and est_seq are test_sizex12x12 arrays
    w, k, l = truth_seq.shape
    true_flows = defaultdict(list)
    est_flows = defaultdict(list)
    res = [0] * (k * l)
    for i in range(w):
        t = truth_seq[i].reshape(-1)
        e = est_seq[i].reshape(-1)
        for j in range(k * l):
            true_flows[j].append(t[j])
            est_flows[j].append(e[j])
    for i in range(k * l):
        if np.linalg.norm(true_flows[i]) == 0.0:
            if np.linalg.norm(est_flows[i]) == 0.0:
                res[i] = 0.0
            else:
                new_true_flows = [x + 1e-7 for x in true_flows]
                res[i] = np.linalg.norm(list(map(operator.sub, new_true_flows[i], est_flows[i]))) / np.linalg.norm(
                    new_true_flows[i])
        else:
            res[i] = np.linalg.norm(list(map(operator.sub, true_flows[i], est_flows[i]))) / np.linalg.norm(
                true_flows[i])
    return res


def RMSE(truth_seq, est_seq):
    res = [0] * truth_seq.shape[0]
    for i in range(truth_seq.shape[0]):
        res[i] = np.sqrt(np.mean(tf.keras.losses.mean_squared_error(truth_seq[i], est_seq[i])))
    return np.mean(res)


def NMAE(truth_seq, est_seq):
    # truth_seq and est_seq are test_sizex12x12 arrays
    res = [0] * truth_seq.shape[0]
    for i in range(truth_seq.shape[0]):
        res[i] = np.sum(np.abs(est_seq[i] - truth_seq[i])) / np.sum(np.abs(truth_seq[i]))
    return np.mean(res)


def load_network_topology(dataset, datapath):
    # initialize graph
    G = nx.DiGraph()
    # load node dataset from csv
    path = os.path.join(datapath, 'topo/{}_node.csv'.format(dataset))
    df = pd.read_csv(path, delimiter=' ')
    for i, row in df.iterrows():
        G.add_node(i, label=row.label, pos=(row.x, row.y))
    # load edge dataset from csv
    path = os.path.join(datapath, 'topo/{}_edge.csv'.format(dataset))
    df = pd.read_csv(path, delimiter=' ')
    # add weight, capacity, delay to edge attributes
    for _, row in df.iterrows():
        i = row.src
        j = row.dest
        G.add_edge(i, j, weight=row.weight,
                   capacity=row.bw,
                   delay=row.delay)
    return G


def load_traindata(args):
    # L_train = np.load(os.path.join(args.datapath, 'vae_data/{}_{}_{}/train_LL.npy'.format(args.dataset,
    #                                                                                                   args.seq_len_x,
    #                                                                                                   args.seq_len_y)))
    # L_val = np.load(os.path.join(args.datapath, 'vae_data/{}_{}_{}/val_LL.npy'.format(args.dataset,
    #                                                                                               args.seq_len_x,
    #                                                                                               args.seq_len_y)))
    X_train = np.load(os.path.join(args.datapath, 'vae_data/{}_{}_{}/train_TM.npy'.format(args.dataset,
                                                                                          args.seq_len_x,
                                                                                          args.seq_len_y)))
    X_val = np.load(os.path.join(args.datapath, 'vae_data/{}_{}_{}/val_TM.npy'.format(args.dataset,
                                                                                      args.seq_len_x,
                                                                                      args.seq_len_y)))
    # A_train = np.load(os.path.join(args.datapath, 'vae_data/{}_{}_{}/train_A.npy'.format(args.dataset,
    #                                                                                                  args.seq_len_x,
    #                                                                                                  args.seq_len_y)))
    # A_val = np.load(os.path.join(args.datapath, 'vae_data/{}_{}_{}/val_A.npy'.format(args.dataset,
    #                                                                                              args.seq_len_x,
    #                                                                                              args.seq_len_y)))

    # X_train = np.reshape(X_train, newshape=(-1, X_train.shape[-1]))
    # X_train = np.reshape(X_train, newshape=(X_train.shape[0], args.nNodes, args.nNodes))

    # X_val = np.reshape(X_val, newshape=(-1, X_val.shape[-1]))
    # X_val = np.reshape(X_val, newshape=(X_val.shape[0], args.nNodes, args.nNodes))

    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)

    return X_train, X_val
    # return X_train, X_val, L_train, L_val, A_train, A_val


def load_test_traffic(args):
    data_path = os.path.join(args.datapath, 'dataset/{}.mat'.format(args.dataset))
    X = loadmat(data_path)['X']
    train, val, test = train_test_split(X, args.dataset)

    test = np.reshape(test, newshape=(test.shape[0], args.nNodes, args.nNodes))

    return test


def train_test_split(X, dataset):
    if 'abilene' in dataset:
        train_size = 3 * 7 * 288  # 3 weeks
        val_size = 288 * 7  # 1 week
        test_size = 288 * 7 * 2  # 2 weeks

    elif 'geant' in dataset:
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

    X_train = X[:train_size]

    X_val = X[train_size:val_size + train_size]

    X_test = X[val_size + train_size: val_size + train_size + test_size]

    if 'abilene' in dataset or 'geant' in dataset or 'brain' in dataset:
        X_train = remove_outliers(X_train)
        X_val = remove_outliers(X_val)

    return X_train, X_val, X_test


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
