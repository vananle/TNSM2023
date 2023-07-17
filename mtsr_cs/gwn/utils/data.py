import os
import pickle
import random as rd

import numpy as np
import torch

rd.seed(42)

from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

homedir = os.path.expanduser('~')
print('Home dir: ', homedir)


class MinMaxScaler_torch:

    def __init__(self, min=None, max=None, device='cuda:0'):
        self.min = min
        self.max = max
        self.device = device

    def fit(self, data):
        self.min = torch.min(data)
        self.max = torch.max(data)

    def transform(self, data):
        _data = data.clone()
        return (_data - self.min) / (self.max - self.min + 1e-8)

    def inverse_transform(self, data):
        return (data * (self.max - self.min + 1e-8)) + self.min


class StandardScaler_torch:

    def __init__(self):
        self.means = 0
        self.stds = 0

    def fit(self, data):
        self.means = torch.mean(data, dim=0)
        self.stds = torch.std(data, dim=0)

    def transform(self, data):
        _data = data.clone()
        data_size = data.size()

        if len(data_size) > 2:
            _data = _data.reshape(-1, data_size[-1])

        _data = (_data - self.means) / (self.stds + 1e-8)

        if len(data_size) > 2:
            _data = _data.reshape(data.size())

        return _data

    def inverse_transform(self, data):
        data_size = data.size()
        if len(data_size) > 2:
            data = data.reshape(-1, data_size[-1])

        data = (data * (self.stds + 1e-8)) + self.means

        if len(data_size) > 2:
            data = data.reshape(data_size)

        return data

    def set_means_stds(self, means, stds, device):
        self.means = torch.Tensor(means)
        self.means = self.means.to(device)
        self.stds = torch.Tensor(stds)
        self.stds = self.stds.to(device)

    def get_mean_stds_numpy(self):
        return self.means.cpu().data.numpy(), self.stds.cpu().data.numpy()


def granularity(data, k):
    if k == 1:
        return np.copy(data)
    else:
        newdata = [np.mean(data[i:i + k], axis=0) for i in range(0, data.shape[0], k)]
        newdata = np.asarray(newdata)
        print('new dataset: ', newdata.shape)
        return newdata


class PartialTrafficDataset(Dataset):

    def __init__(self, dataset, args):
        # save parameters
        self.args = args

        self.type = args.type
        self.out_seq_len = args.out_seq_len
        self.Xtopk = self.np2torch(dataset['Xtopk'])
        self.Ytopk = self.np2torch(dataset['Ytopk'])
        self.Yreal = self.np2torch(dataset['Yreal'])
        self.Xgt = self.np2torch(dataset['Xgt'])
        self.Ygt = self.np2torch(dataset['Ygt'])
        self.Topkindex = dataset['Topkindex']

        self.scaler_topk = StandardScaler_torch()
        self.scaler_topk.set_means_stds(dataset['Scaler_topk'][0], dataset['Scaler_topk'][1], args.device)

        self.nsample, self.len_x, self.nflows, self.nfeatures = self.Xtopk.shape

        # get valid start indices for sub-series
        self.indices = self.get_indices()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        x_top_k = self.Xtopk[t]
        y_top_k = self.Ytopk[t]
        y_real = self.Yreal[t]
        xgt = self.Xgt[t]
        ygt = self.Ygt[t]
        sample = {'x_top_k': x_top_k, 'y_top_k': y_top_k, 'x_gt': xgt, 'y_gt': ygt, 'y_real': y_real}
        return sample

    def transform(self, X):
        return self.scaler_topk.transform(X)

    def inverse_transform(self, X):
        return self.scaler_topk.inverse_transform(X)

    def np2torch(self, X):
        X = torch.Tensor(X)
        if torch.cuda.is_available():
            X = X.to(self.args.device)
        return X

    def get_indices(self):
        indices = np.arange(self.nsample)
        return indices


def load_raw(args):
    # load ground truth

    data_path = os.path.join(args.datapath, 'dataset/{}.mat'.format(args.dataset))
    print(data_path)

    X = loadmat(data_path)['X']
    if len(X.shape) > 2:
        X = np.reshape(X, newshape=(X.shape[0], -1))

    print('dataset {} shape {} mean {}'.format(args.dataset, X.shape, X.mean()))

    return X


def np2torch(X, device):
    X = torch.Tensor(X)
    if torch.cuda.is_available():
        X = X.to(device)
    return X


def topk_train(Xscaledtopk, Xtopk, X, oX, t, args):
    x_scaled_topk = torch.clone(Xscaledtopk[t:t + args.seq_len_x])
    x_scaled_topk = x_scaled_topk.unsqueeze(dim=-1)  # add feature dim [seq_x, n, 1]

    y_topk = torch.max(Xtopk[t + args.seq_len_x:t + args.seq_len_x + args.seq_len_y], dim=0)[0]
    y_topk = y_topk.reshape(1, -1)

    y_real = torch.max(X[t + args.seq_len_x:t + args.seq_len_x + args.seq_len_y], dim=0)[0]
    y_real = y_real.reshape(1, -1)

    # Data for doing traffic engineering
    x_gt = torch.clone(oX[t:t + args.seq_len_x])
    y_gt = torch.clone(oX[t + args.seq_len_x: t + args.seq_len_x + args.seq_len_y])

    return x_scaled_topk, y_topk, y_real, x_gt, y_gt


def data_preprocessing(data, topk_index, args, gen_times=5, scaler_top_k=None):
    n_timesteps, n_series = data.shape

    # original dataset with granularity k = 1
    oX = np.copy(data)
    oX = np2torch(oX, args.device)

    # Obtain dataset with different granularity
    # X = granularity(dataset, args.k)
    # test with k = 1
    X = np.copy(data)

    # Obtain dataset with topk flows
    X_top_k = np.copy(X[:, topk_index])

    # Load dataset to devices
    X = np2torch(X, args.device)
    X_top_k = np2torch(X_top_k, args.device)

    # scaling dataset
    scaler = StandardScaler_torch()
    if scaler_top_k is None:
        scaler.fit(X_top_k)
    else:
        scaler.set_means_stds(scaler_top_k[0], scaler_top_k[1], args.device)

    X_scaled_top_k = scaler.transform(X_top_k)

    len_x = args.seq_len_x
    len_y = args.seq_len_y

    scaler_means, scaler_std = scaler.get_mean_stds_numpy()
    dataset = {'Xtopk': [], 'Ytopk': [], 'Xgt': [], 'Ygt': [], 'Yreal': [],
               'Topkindex': topk_index, 'Scaler_topk': [scaler_means, scaler_std], 'device': args.device}

    skip = 1
    start_idx = 0
    for _ in range(gen_times):
        print('Gen {}, start_idx {}'.format(_, start_idx))
        for t in range(start_idx, n_timesteps - len_x - len_y, len_x):
            if args.fs == 'train':

                x_scaled_topk, y_topk, y_real, x_gt, y_gt = topk_train(Xscaledtopk=X_scaled_top_k,
                                                                       Xtopk=X_top_k,
                                                                       X=X, oX=oX, t=t, args=args)

            else:
                raise RuntimeError('No flow selection!')

            if torch.max(x_gt) <= 0.0 or torch.max(y_gt) <= 0.0:
                continue

            dataset['Xtopk'].append(x_scaled_topk)  # [sample, len_x, k, 1]
            dataset['Ytopk'].append(y_topk)  # [sample, 1, k]
            dataset['Yreal'].append(y_real)  # [sample, 1, k]
            dataset['Xgt'].append(x_gt)
            dataset['Ygt'].append(y_gt)

        start_idx += skip

    dataset['Xtopk'] = torch.stack(dataset['Xtopk'], dim=0)
    dataset['Ytopk'] = torch.stack(dataset['Ytopk'], dim=0)
    dataset['Yreal'] = torch.stack(dataset['Yreal'], dim=0)
    dataset['Xgt'] = torch.stack(dataset['Xgt'], dim=0)
    dataset['Ygt'] = torch.stack(dataset['Ygt'], dim=0)

    dataset['Xtopk'] = dataset['Xtopk'].cpu().data.numpy()
    dataset['Ytopk'] = dataset['Ytopk'].cpu().data.numpy()
    dataset['Yreal'] = dataset['Yreal'].cpu().data.numpy()
    dataset['Xgt'] = dataset['Xgt'].cpu().data.numpy()
    dataset['Ygt'] = dataset['Ygt'].cpu().data.numpy()

    print('   Xtopk: ', dataset['Xtopk'].shape)
    print('   Ytopk: ', dataset['Ytopk'].shape)
    print('   Yreal: ', dataset['Yreal'].shape)
    print('   Xgt: ', dataset['Xgt'].shape)
    print('   Ygt: ', dataset['Ygt'].shape)
    print('   Topkindex: ', dataset['Topkindex'].shape)

    return dataset


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
    elif 'brain' in dataset:
        train_size = 1440 * 3  # 3 days
        val_size = 1440  # 1 day
        test_size = 1440 * 2  # 2 days
    elif 'uninett' in dataset:  # granularity: 5 min
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

    print('Raw dataset:')
    print('X_train: ', X_train.shape)
    print('X_val: ', X_val.shape)
    print('X_test: ', X_test.shape)

    return X_train, X_val, X_test


def get_dataloader(args):
    # loading dataset

    args.datapath = os.path.join(homedir, args.datapath)

    X = load_raw(args)
    total_timesteps, total_series = X.shape

    stored_path = os.path.join(args.datapath, 'cs_data/cs_{}_{}_{}_{}_{}/'.format(args.dataset, args.seq_len_x,
                                                                                  args.seq_len_y, args.mon_rate,
                                                                                  args.fs))
    if not os.path.exists(stored_path):
        os.makedirs(stored_path)

    saved_train_path = os.path.join(stored_path, 'train.pkl')
    saved_val_path = os.path.join(stored_path, 'val.pkl')
    saved_test_path = os.path.join(stored_path, 'test.pkl')
    if not os.path.exists(saved_train_path) or not os.path.exists(saved_val_path) or not os.path.exists(
            saved_test_path):
        train, val, test = train_test_split(X, args.dataset)
        # obtain topk largest flows index from training set
        means = np.mean(train, axis=0)
        top_k_index = np.argsort(means)[::-1]
        top_k_index = top_k_index[:int(args.mon_rate * total_series / 100)]

        print('Data preprocessing: TRAINSET')
        trainset = data_preprocessing(data=train, topk_index=top_k_index, args=args, gen_times=10, scaler_top_k=None)
        train_scaler = trainset['Scaler_topk']
        with open(saved_train_path, 'wb') as fp:
            pickle.dump(trainset, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()

        print('Data preprocessing: VALSET')
        valset = data_preprocessing(data=val, topk_index=top_k_index, args=args, gen_times=10,
                                    scaler_top_k=train_scaler)
        with open(saved_val_path, 'wb') as fp:
            pickle.dump(valset, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()

        print('Data preprocessing: TESTSET')
        testset = data_preprocessing(data=test, topk_index=top_k_index,
                                     args=args, gen_times=1, scaler_top_k=train_scaler)

        with open(saved_test_path, 'wb') as fp:
            pickle.dump(testset, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()
    else:
        print('Load saved dataset from {}'.format(stored_path))
        with open(saved_test_path, 'rb') as fp:
            testset = pickle.load(fp)
            fp.close()

        args.device = testset['device']

        if args.test:
            trainset, valset = None, None
        else:
            with open(saved_train_path, 'rb') as fp:
                trainset = pickle.load(fp)
                fp.close()
            with open(saved_val_path, 'rb') as fp:
                valset = pickle.load(fp)
                fp.close()

    test_set = PartialTrafficDataset(testset, args=args)
    test_loader = DataLoader(test_set,
                             batch_size=args.test_batch_size,
                             shuffle=False)

    if args.test:  # Only load testing set
        train_loader = None
        val_loader = None
    else:
        # Training set
        train_set = PartialTrafficDataset(trainset, args=args)
        train_loader = DataLoader(train_set,
                                  batch_size=args.train_batch_size,
                                  shuffle=True)

        # validation set
        val_set = PartialTrafficDataset(valset, args=args)
        val_loader = DataLoader(val_set,
                                batch_size=args.val_batch_size,
                                shuffle=False)

    return train_loader, val_loader, test_loader, total_timesteps, total_series
