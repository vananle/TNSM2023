import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class MinMaxScaler:
    def __init__(self, copy=True):
        self.min = np.inf
        self.max = -np.inf
        self.copy = copy
        self.fit_data = False

    def fit(self, data):
        self.min = np.min(data) if np.min(data) < self.min else self.min
        self.max = np.max(data) if np.max(data) > self.max else self.max
        self.fit_data = True

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def transform(self, data):
        if not self.fit_data:
            raise RuntimeError('Fit data first!')

        if self.copy:
            _data = np.copy(data)
        else:
            _data = data

        scaled_data = (_data - self.min) / (self.max - self.min + 1e-10)
        return scaled_data

    def inverse_transform(self, data):
        if not self.fit_data:
            raise RuntimeError('Fit data first!')

        if self.copy:
            _data = np.copy(data)
        else:
            _data = data

        inverse_data = _data * (self.max - self.min + 1e-10) + self.min
        return inverse_data


def load_raw_data(args):
    path = os.path.join(args.data_folder, f'{args.dataset}.npz')

    all_data = np.load(path)
    data = all_data['traffic_demands']

    if len(data.shape) > 2:
        data = np.reshape(data, newshape=(data.shape[0], -1))

    # calculate num node
    T, F = data.shape
    N = int(np.sqrt(F))
    args.num_node = N
    args.num_flow = F
    # print('Data shape', data.shape)

    data[data <= 0] = 1e-4
    data[data == np.nan] = 1e-4
    # Train-test split

    total_steps = get_data_size(dataset=args.dataset, data=data)
    data_traffic = data[:total_steps]

    train_size = int(0.7 * total_steps)
    val_size = int(0.1 * total_steps)

    train_df, val_df, test_df = data_traffic[0:train_size], data_traffic[train_size:train_size + val_size], \
        data_traffic[train_size + val_size:]  # total dataset

    return train_df, val_df, test_df


def get_data_size(dataset, data):
    if 'abilene' in dataset:
        return 3000  # use 3000 traffic matrices of abilene dataset > 10 days
    elif 'geant' in dataset:
        return 1000  # use 1000 traffic matrices of geant dataset > 10 days
    else:
        return data.shape[0]


def data_split(args):
    train_df, val_df, test_df = load_raw_data(args)

    sc = MinMaxScaler(copy=True)
    sc.fit(train_df)

    train_scaled = sc.transform(train_df)
    val_scaled = sc.transform(val_df)
    test_scaled = sc.transform(test_df)

    # Converting the time series to samples
    def create_dataset(data, data_scaled, input_len=12, predict_len=1):
        X, y = [], []
        X_gt, y_gt = [], []
        i0 = input_len
        close_indices = np.arange(input_len, dtype=np.int32) - input_len
        predict_indices = np.arange(predict_len)
        for i in tqdm(range(i0, len(data) - predict_len)):
            feature = data_scaled[i + close_indices]
            label = np.max(data_scaled[i + predict_indices], axis=0)
            X.append(feature)
            y.append(label)

            feature_gt = data[i + close_indices]
            label_gt = data[i + predict_indices]
            X_gt.append(feature_gt)
            y_gt.append(label_gt)

        return np.array(X), np.array(y), np.array(X_gt), np.array(y_gt)

    x_train, y_train, xgt_train, ygt_train = create_dataset(data=train_df, data_scaled=train_scaled,
                                                            input_len=args.input_len,
                                                            predict_len=args.predict_len)
    x_val, y_val, xgt_val, ygt_val = create_dataset(data=val_df, data_scaled=val_scaled,
                                                    input_len=args.input_len,
                                                    predict_len=args.predict_len)
    x_test, y_test, xgt_test, ygt_test = create_dataset(data=test_df, data_scaled=test_scaled,
                                                        input_len=args.input_len,
                                                        predict_len=args.predict_len)

    print("x_train_shape", x_train.shape)
    print("x_val_shape", x_val.shape)
    print("x_test_shape", x_test.shape)
    print("y_train", y_train.shape)
    print("y_val", y_val.shape)
    print("y_test", y_test.shape)

    data = {
        'train/x': x_train,
        'val/x': x_val,
        'test/x': x_test,
        'train/y': y_train,
        'val/y': y_val,
        'test/y': y_test,
        'scaler': sc,
        'train/x_gt': xgt_train,
        'val/x_gt': xgt_val,
        'test/x_gt': xgt_test,
        'train/y_gt': ygt_train,
        'val/y_gt': ygt_val,
        'test/y_gt': ygt_test,
    }

    return data


def get_monitoring_index(data, prev_id_mon_flow, is_first_step, args):
    """

    @param data: the raw traffic data
    @param prev_id_mon_flow: the indices of monitored flows of a previous routing round
    @param is_first_step: is this the first step
    @param args:
    @return: the index of monitored flow
    """
    mon_method = args.mon_method

    if mon_method == 'random':
        num_mon_flow = int(args.mon_per * args.num_flow)
        args.num_mon_flow = num_mon_flow
        if np.random.uniform(0, 1) > 0.7 or is_first_step:
            id_mon_flow = np.random.choice(a=np.arange(0, args.num_flow), size=num_mon_flow, replace=False)
        else:
            id_mon_flow = copy.deepcopy(prev_id_mon_flow)
    elif mon_method == 'topk':
        num_mon_flow = int(args.mon_per * args.num_flow)
        args.num_mon_flow = num_mon_flow
        mean_traffic = np.mean(data, axis=0)
        id_mon_flow = np.argsort(mean_traffic)[::-1][:num_mon_flow]
    elif mon_method == 'topk_random':

        id_mon_flow = copy.deepcopy(prev_id_mon_flow)
        num_mon_flow = int(args.mon_per * args.num_flow)
        args.num_mon_flow = num_mon_flow

        if is_first_step:
            mean_traffic = np.mean(data, axis=0)
            id_mon_flow = np.argsort(mean_traffic)[::-1][:num_mon_flow]
        else:

            num_new_id_mon_flow = int(0.5 * num_mon_flow)
            for i in range(int(num_new_id_mon_flow)):
                rand_idx = np.random.randint(0, args.num_flow)
                old_id_mon_flow = id_mon_flow[0:(num_mon_flow - num_new_id_mon_flow + i)]
                while rand_idx in old_id_mon_flow:
                    rand_idx = np.random.randint(0, args.num_flow)

                id_mon_flow[num_mon_flow - num_new_id_mon_flow + i] = rand_idx
    elif mon_method == 'topk_per_node':
        num_mon_flow = int(args.mon_per * args.num_node) * args.num_node
        args.num_mon_flow = num_mon_flow

        _data = np.reshape(data, newshape=(data.shape[0], args.num_node, args.num_node))

        id_mon_flow = []
        for j in range(args.num_node):
            data_i = _data[:, j, :]

            mean_traffic = np.mean(data_i, axis=0)
            _index = np.argsort(mean_traffic)[::-1][:int(args.topk * args.num_node)]
            id_mon_flow.append(_index + j * args.num_node)
    else:
        raise NotImplementedError

    return id_mon_flow


def data_split_cs(args):
    train_df, val_df, test_df = load_raw_data(args)

    sc = MinMaxScaler(copy=True)
    sc.fit(train_df)

    train_scaled = sc.transform(train_df)
    val_scaled = sc.transform(val_df)
    test_scaled = sc.transform(test_df)

    # Converting the time series to samples
    def create_dataset(data, data_scaled, input_len=12, predict_len=1):
        X, y = [], []
        X_gt, y_gt, y_gt_max = [], [], []
        topk_index = []
        i0 = input_len
        close_indices = np.arange(input_len, dtype=np.int32) - input_len
        predict_indices = np.arange(predict_len)
        prev_id_mon_flow = None
        for i in tqdm(range(i0, len(data) - predict_len)):
            feature_gt = data[i + close_indices]
            label_gt = data[i + predict_indices]
            label_gt_max = np.max(data[i + predict_indices], axis=0)

            id_mon_flow = get_monitoring_index(data=feature_gt, prev_id_mon_flow=prev_id_mon_flow,
                                               is_first_step=True if i == i0 else False,
                                               args=args)

            id_mon_flow = np.array(id_mon_flow).flatten()
            topk_index.append(id_mon_flow)
            X_gt.append(feature_gt)
            y_gt.append(label_gt)
            y_gt_max.append(label_gt_max)

            feature = data_scaled[i + close_indices]
            feature = feature[:, id_mon_flow]
            future_traffic = data_scaled[i + predict_indices]
            future_traffic = future_traffic[:, id_mon_flow]
            label = np.max(future_traffic, axis=0)
            X.append(feature)
            y.append(label)

            prev_id_mon_flow = copy.deepcopy(id_mon_flow)

        return np.array(X), np.array(y), np.array(X_gt), np.array(y_gt), np.array(y_gt_max), np.array(topk_index)

    x_train, y_train, xgt_train, ygt_train, ygt_max_train, mon_index_train = create_dataset(data=train_df,
                                                                                            data_scaled=train_scaled,
                                                                                            input_len=args.input_len,
                                                                                            predict_len=args.predict_len)
    x_val, y_val, xgt_val, ygt_val, ygt_max_val, mon_index_val = create_dataset(data=val_df, data_scaled=val_scaled,
                                                                                input_len=args.input_len,
                                                                                predict_len=args.predict_len)
    x_test, y_test, xgt_test, ygt_test, ygt_max_test, mon_index_test = create_dataset(data=test_df,
                                                                                      data_scaled=test_scaled,
                                                                                      input_len=args.input_len,
                                                                                      predict_len=args.predict_len)

    print("x_train_shape", x_train.shape)
    print("x_val_shape", x_val.shape)
    print("x_test_shape", x_test.shape)
    print("y_train", y_train.shape)
    print("y_val", y_val.shape)
    print("y_test", y_test.shape)
    print("ygt_max_train", ygt_max_train.shape)
    print("ygt_max_val", ygt_max_val.shape)
    print("ygt_max_test", ygt_max_test.shape)
    print("mon_index_train", mon_index_train.shape)
    print("mon_index_val", mon_index_val.shape)
    print("mon_index_test", mon_index_test.shape)

    data = {
        'train/x': x_train,
        'val/x': x_val,
        'test/x': x_test,
        'train/y': y_train,
        'val/y': y_val,
        'test/y': y_test,
        'scaler': sc,
        'train/x_gt': xgt_train,
        'val/x_gt': xgt_val,
        'test/x_gt': xgt_test,
        'train/y_gt': ygt_train,
        'val/y_gt': ygt_val,
        'test/y_gt': ygt_test,
        'train/y_gt_max': ygt_max_train,
        'val/y_gt_max': ygt_max_val,
        'test/y_gt_max': ygt_max_test,
        'train/mon_index': mon_index_train,
        'val/mon_index': mon_index_val,
        'test/mon_index': mon_index_test,
    }

    return data


def data_split_vae(args):
    train_df, val_df, test_df = load_raw_data(args)

    sc = MinMaxScaler(copy=True)
    sc.fit(train_df)

    train_scaled = sc.transform(train_df)
    val_scaled = sc.transform(val_df)
    test_scaled = sc.transform(test_df)

    def create_dataset(data, data_scaled, input_len=12, predict_len=1):
        X, y = [], []
        X_gt, y_gt = [], []
        i0 = input_len
        close_indices = np.arange(input_len, dtype=np.int32) - input_len
        predict_indices = np.arange(predict_len)
        for i in tqdm(range(i0, len(data) - predict_len)):
            feature = data_scaled[i + close_indices]
            label = np.max(data_scaled[i + predict_indices], axis=0)
            X.append(feature)
            y.append(label)

            feature_gt = data[i + close_indices]
            label_gt = data[i + predict_indices]
            X_gt.append(feature_gt)
            y_gt.append(label_gt)

        return np.array(X), np.array(y), np.array(X_gt), np.array(y_gt)

    x_train, y_train, xgt_train, ygt_train = create_dataset(data=train_df, data_scaled=train_scaled,
                                                            input_len=args.input_len,
                                                            predict_len=args.predict_len)
    x_val, y_val, xgt_val, ygt_val = create_dataset(data=val_df, data_scaled=val_scaled,
                                                    input_len=args.input_len,
                                                    predict_len=args.predict_len)
    x_test, y_test, xgt_test, ygt_test = create_dataset(data=test_df, data_scaled=test_scaled,
                                                        input_len=args.input_len,
                                                        predict_len=args.predict_len)

    # Converting the time series to samples

    data = {
        'train/gt': train_df,
        'val/gt': val_df,
        'test/gt': test_df,
        'train/scaled': train_scaled,
        'val/scaled': val_scaled,
        'test/scaled': test_scaled,
        'scaler': sc,
        'train/x_gt': xgt_train,
        'val/x_gt': xgt_val,
        'test/x_gt': xgt_test,
        'train/y_gt': ygt_train,
        'val/y_gt': ygt_val,
        'test/y_gt': ygt_test
    }

    return data


def get_dataloader(args):
    if args.method == 'mtsr':
        data = data_split(args=args)

    elif args.method == 'mtsr_cs' or args.method == 'mtsr_nocs':
        data = data_split_cs(args=args)
    else:
        raise NotImplementedError

    x_train = torch.from_numpy(data['train/x']).to(dtype=torch.float32, device=args.device)
    y_train = torch.from_numpy(data['train/y']).to(dtype=torch.float32, device=args.device)
    train_loader = DataLoader(list(zip(x_train, y_train)), shuffle=False,
                              batch_size=args.train_batch_size)

    x_val = torch.from_numpy(data['val/x']).to(dtype=torch.float32, device=args.device)
    y_val = torch.from_numpy(data['val/y']).to(dtype=torch.float32, device=args.device)
    val_loader = DataLoader(list(zip(x_val, y_val)), shuffle=False, batch_size=args.val_batch_size)

    x_test = torch.from_numpy(data['test/x']).to(dtype=torch.float32, device=args.device)
    y_test = torch.from_numpy(data['test/y']).to(dtype=torch.float32, device=args.device)
    test_loader = DataLoader(list(zip(x_test, y_test)), shuffle=False,
                             batch_size=args.test_batch_size)

    data.update({
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
    })

    return data


def get_data_vae(args):
    return data_split_vae(args)
