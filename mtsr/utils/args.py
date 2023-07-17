import argparse
import os
import random

import numpy as np
import scipy.sparse as sp
import torch

ADJ_CHOICES = ['scalap', 'normlap', 'symnadj', 'transition', 'identity']


def set_random_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_args():
    # create argument parser
    parser = argparse.ArgumentParser()

    # parameter for dataset
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='abilene',
                        choices=['abilene', 'geant', 'sdn', 'germany'],
                        help='Dataset, (default abilene)')
    parser.add_argument('--data_folder', type=str, default='../data')
    parser.add_argument('--tensorboard_folder', type=str, default='../logs/mtsr/')
    parser.add_argument('--csv_folder', type=str, default='../data/csv')
    parser.add_argument('--model_folder', type=str, default='../logs/mtsr/')

    parser.add_argument('--type', type=str, default='p2', choices=['p1', 'p2', 'p3'],
                        help='problem formulation (default p2)')
    parser.add_argument('--trunk', type=int, default=3, help='trunk for p3 problem (default 3)')
    parser.add_argument('--k', type=int, default=1, help='granularity scale', choices=[1, 2, 3])

    # Method
    parser.add_argument('--method', type=str, default='mtsr', choices=['mtsr', 'ls2sr', 'mtsr_cs', 'ls2sr_cs', 'vae_sr',
                                                                       'cfr_rl'])
    parser.add_argument('--mon_method', type=str, default='random', choices=['random', 'topk', 'topk_random',
                                                                             'topk_per_node'])
    parser.add_argument('--mon_per', type=float, default=0.1)

    # Model
    # Graph
    parser.add_argument('--model', type=str, default='gwn', help='Model default GWN',
                        choices=['gwn', 'lstm', 'fbf_lstm', 'gru', 'fbf_gru', 'stgcn', 'dcrnn', 'mtgnn', 'vae'])

    # vae
    parser.add_argument('--latent_dim', type=int, default=16, help='latent_dim')

    # Wavenet
    parser.add_argument('--input_len', type=int, default=12,
                        help='input length default 64')
    parser.add_argument('--predict_len', type=int, default=12, choices=[3, 6, 9, 12, 24],
                        help='routing cycle 12')

    parser.add_argument('--blocks', type=int, default=5, help='')
    parser.add_argument('--layers', type=int, default=3, help='')
    parser.add_argument('--hidden', type=int, default=32, help='Number of channels for internal conv')
    parser.add_argument('--kernel_size', type=int, default=4, help='kernel_size for internal conv')
    parser.add_argument('--stride', type=int, default=4, help='stride for internal conv')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')

    # loss
    parser.add_argument('--loss_fn', type=str, default='mse', choices=['mse', 'mae'])

    # training
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--num_epochs', type=int, default=300, help='')
    parser.add_argument('--clip', type=int, default=1, help='Gradient Clipping')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--patience', type=int, default=50, help='quit if no improvement after this many iterations')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--plot', action='store_true')

    # parameter for test_routing
    parser.add_argument('--te_alg', type=str,
                        choices=['None', 'ls2sr', 'p0', 'p1', 'p2', 'p3', 'or', 'srls', 'ob', 'sp'],
                        default='None')

    parser.add_argument('--use_gt', action='store_true')
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--nrun', type=int, default=1)

    # get args
    args = parser.parse_args()
    return args


def args_adjust(args):
    if 'abilene' in args.dataset:
        args.num_node = 12
        args.num_flow = args.num_node * args.num_node
        args.day_size = 288
    elif 'geant' in args.dataset:
        args.num_node = 22
        args.num_flow = args.num_node * args.num_node
        args.day_size = 96
    elif 'sdn' in args.dataset:
        args.num_node = 14
        args.num_flow = args.num_node * args.num_node
        args.day_size = 1440
    elif 'germany' in args.dataset:
        args.num_node = 50
        args.num_flow = args.num_node * args.num_node
        args.day_size = 288
    else:
        raise ValueError('Dataset not found!')

    if args.type == 'p1':
        args.output_len = args.predict_len
    elif args.type == 'p2':
        args.output_len = 1
    elif args.type == 'p3':
        if args.predict_len % args.trunk != 0:
            args.predict_len = int(args.predict_len / args.trunk) * args.trunk
        args.output_len = args.trunk

    if args.input_len == 3:
        args.blocks = 1
        args.layers = 2
        args.kernel_size = 2
        args.stride = 2
    elif args.input_len == 6:
        args.blocks = 2
        args.layers = 2
        args.kernel_size = 2
        args.stride = 2
    elif args.input_len == 9:
        args.blocks = 2
        args.layers = 3
        args.kernel_size = 2
        args.stride = 2
    elif args.input_len == 12:
        args.blocks = 4
        args.layers = 2
        args.kernel_size = 2
        args.stride = 2
    elif args.input_len == 15:
        args.blocks = 2
        args.layers = 3
        args.kernel_size = 2
        args.stride = 2
    elif args.input_len == 18:
        args.blocks = 2
        args.layers = 4
        args.kernel_size = 2
        args.stride = 2
    elif args.input_len == 21:
        args.blocks = 2
        args.layers = 4
        args.kernel_size = 2
        args.stride = 2
    elif args.input_len == 24:
        args.blocks = 2
        args.layers = 4
        args.kernel_size = 2
        args.stride = 2
    elif args.input_len == 27:
        args.blocks = 2
        args.layers = 4
        args.kernel_size = 2
        args.stride = 2
    elif args.input_len == 30:
        args.blocks = 4
        args.layers = 2
        args.kernel_size = 3
        args.stride = 3
    elif args.input_len == 36:
        args.blocks = 3
        args.layers = 1
        args.kernel_size = 4
        args.stride = 4
    elif args.input_len == 48:
        args.blocks = 4
        args.layers = 2
        args.kernel_size = 4
        args.stride = 4
    elif args.input_len == 60:
        args.blocks = 6
        args.layers = 2
        args.kernel_size = 4
        args.stride = 4
    elif args.input_len == 72:
        args.blocks = 5
        args.layers = 2
        args.kernel_size = 4
        args.stride = 4
    elif args.input_len == 96:
        args.blocks = 5
        args.layers = 2
        args.kernel_size = 4
        args.stride = 4
    else:
        raise NotImplemented('input_len!')

    in_dim = 1
    args.in_dim = in_dim

    path = os.path.join(args.data_folder, f'{args.dataset}.npz')
    data = np.load(path)

    m_adj = data['adj_mx']
    if args.model == 'stgcn':
        m_adj = calculate_normalized_laplacian(m_adj).astype(np.float32).todense()
        args.m_adj = torch.from_numpy(m_adj).to(args.device)
    else:
        args.m_adj = m_adj

    set_random_seed(args)
    update_num_mon_flow(args)

    return args


def update_num_mon_flow(args):
    if args.method == 'mtsr':
        args.mon_per = 1.0
        args.num_mon_flow = args.num_flow
    elif 'cs' in args.method:
        if args.mon_method == 'random':
            num_mon_flow = int(args.mon_per * args.num_flow)
            args.num_mon_flow = num_mon_flow
        elif args.mon_method == 'topk':
            num_mon_flow = int(args.mon_per * args.num_flow)
            args.num_mon_flow = num_mon_flow
        elif args.mon_method == 'topk_random':
            num_mon_flow = int(args.mon_per * args.num_flow)
            args.num_mon_flow = num_mon_flow
        elif args.mon_method == 'topk_per_node':
            num_mon_flow = int(args.mon_per * args.num_node) * args.num_node
            args.num_mon_flow = num_mon_flow
        else:
            raise NotImplemented
    else:
        raise NotImplemented

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


def print_args(args):
    print('-------------------------------------')
    print('[+] Time-series recovering experiment')
    if args.test:
        print('|--- Run Test')
    else:
        print('|--- Run Train')

    print('---------------------------------------------------------')
    print('[+] Time-series prediction experiment')
    print('---------------------------------------------------------')
    print('    - dataset                :', args.dataset)
    print('    - num_flow               :', args.num_flow)
    print('---------------------------------------------------------')
    print('    - model                  :', args.model)
    print('    - wn_blocks              :', args.blocks)
    print('    - wn_layers              :', args.layers)
    print('    - hidden                 :', args.hidden)
    print('    - kernel_size            :', args.kernel_size)
    print('    - stride                 :', args.stride)
    print('----------------------------')
    print('    - type                   :', args.type)
    print('    - input_len              :', args.input_len)
    print('    - predict_len            :', args.predict_len)
    print('    - output_len            :', args.output_len)
    print('---------------------------------------------------------')
    print('    - device                 :', args.device)
    print('    - train_batch_size       :', args.train_batch_size)
    print('    - val_batch_size         :', args.val_batch_size)
    print('    - test_batch_size        :', args.test_batch_size)
    print('    - epochs                 :', args.num_epochs)
    print('    - learning_rate          :', args.lr)
    print('    - patience               :', args.patience)
    print('---------------------------------------------------------')
    print('    - te_alg                 :', args.te_alg)
    print('---------------------------------------------------------')
