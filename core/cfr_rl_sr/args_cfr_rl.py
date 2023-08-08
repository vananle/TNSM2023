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
                        choices=['abilene', 'geant', 'sdn'],
                        help='Dataset, (default abilene)')
    parser.add_argument('--data_folder', type=str, default='../data')
    parser.add_argument('--tensorboard_folder', type=str, default='../logs/cfr')
    parser.add_argument('--csv_folder', type=str, default='../data/csv')
    parser.add_argument('--model_folder', type=str, default='../logs/cfr/model')

    # neural network args

    parser.add_argument('--num_agents', type=int, default=5)
    parser.add_argument('--num_iter', type=int, default=2)
    # neural network args
    parser.add_argument('--scale', type=int, default=100)
    parser.add_argument('--max_step', type=int, default=1000)
    parser.add_argument('--initial_learning_rate', type=float, default=0.0001)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.96)
    parser.add_argument('--learning_rate_decay_step', type=int, default=5)
    parser.add_argument('--moving_average_decay', type=float, default=0.9999)
    parser.add_argument('--entropy_weight', type=float, default=0.1)
    parser.add_argument('--save_step', type=int, default=100)
    parser.add_argument('--max_to_keep', type=int, default=100)
    parser.add_argument('--Conv2D_out', type=int, default=128)
    parser.add_argument('--Dense_out', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--logit_clipping', type=int, default=10)

    parser.add_argument('--version', type=str, default='TE_v2')
    parser.add_argument('--project_name', type=str, default='CFR-RL')
    parser.add_argument('--method', type=str, default='actor_critic')
    parser.add_argument('--model_type', type=str, default='Conv')

    parser.add_argument('--tm_history', type=int, default=1)
    parser.add_argument('--routing_cycle', type=int, default=12)
    parser.add_argument('--max_moves', type=float, default=10,
                        help='percentage of number of flows to be considered as critical flows (default 10%)')
    parser.add_argument('--baseline', type=str, default='avg')
    parser.add_argument('--eval_methods', type=str, default='topk',
                        choices=['cfr_rl', 'cfr_topk', 'topk', 'optimal', 'sp'])

    # training
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--timeout', type=float, default=10.0)

    # get args
    args = parser.parse_args()
    return args


def args_adjust(args):
    if 'abilene' in args.dataset:
        args.num_node = 12
        args.num_flow = args.num_node*args.num_node
        args.day_size = 288
    elif 'geant' in args.dataset:
        args.num_node = 22
        args.num_flow = args.num_node*args.num_node
        args.day_size = 96
    elif 'sdn' in args.dataset:
        args.num_node = 14
        args.num_flow = args.num_node*args.num_node
        args.day_size = 1440
    elif 'germany' in args.dataset:
        args.num_node = 50
        args.num_flow = args.num_node*args.num_node
        args.day_size = 288
    else:
        raise ValueError('Dataset not found!')

    args.max_step = args.max_step * args.scale
    args.learning_rate_decay_step = args.learning_rate_decay_step * args.scale
    args.save_step = args.save_step

    set_random_seed(args)

    args.model_saved_dir = os.path.join(args.model_folder, 'cfr_rl')

    return args


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
