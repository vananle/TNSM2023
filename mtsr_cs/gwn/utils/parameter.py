import argparse

ADJ_CHOICES = ['scalap', 'normlap', 'symnadj', 'transition', 'identity']

import os

homedir = os.path.expanduser('~')


def get_args():
    # create argument parser
    parser = argparse.ArgumentParser()

    # parameter for dataset
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='abilene_tm',
                        choices=['abilene_tm', 'geant_tm', 'brain_tm', 'uninett_tm', 'renater_tm', 'geant3_tm'],
                        help='Dataset, (default abilene_tm)')
    parser.add_argument('--intv', type=int, default=5, help='Dataset, (default abilene_tm)')

    parser.add_argument('--mon_rate', type=int, default=5)
    parser.add_argument('--cs', type=int, default=1, choices=[0, 1])

    parser.add_argument('--datapath', type=str, default='thesis_data/dataset/')
    parser.add_argument('--type', type=str, default='p2', choices=['p1', 'p2', 'p3'],
                        help='problem formulation (default p2)')
    parser.add_argument('--trunk', type=int, default=3, help='trunk for p3 problem (default 3)')
    parser.add_argument('--k', type=int, default=1, help='granularity scale', choices=[1, 2, 3])

    parser.add_argument('--tod', action='store_true')
    parser.add_argument('--ma', action='store_true')
    parser.add_argument('--mx', action='store_true')

    # Model
    # Graph
    parser.add_argument('--model', type=str, default='gwn')
    parser.add_argument('--adjdata', type=str, default='../../dataset/dataset/sensor_graph/adj_mx.pkl',
                        help='adj dataset path')
    parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type (default doubletransition)',
                        choices=ADJ_CHOICES)
    parser.add_argument('--do_graph_conv', action='store_true',
                        help='whether to add graph convolution layer')
    parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
    parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
    parser.add_argument('--randomadj', action='store_true',
                        help='whether random initialize adaptive adj')
    parser.add_argument('--apt_size', default=10, type=int, help='default 10')

    # Wavenet
    parser.add_argument('--seq_len_x', type=int, default=36, choices=[6, 12, 18, 24, 30, 36, 48],
                        help='input length default 64')
    parser.add_argument('--seq_len_y', type=int, default=36, choices=[6, 12, 18, 24, 30, 36, 48],
                        help='routing cycle 12')

    parser.add_argument('--dilation_channels', type=int, default=32, help='inputs dimension (default 32)')
    parser.add_argument('--residual_channels', type=int, default=32, help='inputs dimension')
    parser.add_argument('--skip_channels', type=int, default=64, help='inputs dimension')
    parser.add_argument('--end_channels', type=int, default=128, help='inputs dimension')

    parser.add_argument('--blocks', type=int, default=4, help='')
    parser.add_argument('--layers', type=int, default=3, help='')
    parser.add_argument('--hidden', type=int, default=32, help='Number of channels for internal conv')
    parser.add_argument('--kernel_size', type=int, default=2, help='kernel_size for internal conv')
    parser.add_argument('--stride', type=int, default=2, help='stride for internal conv')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--n_obs', default=None, help='Only use this many observations. For unit testing.')
    parser.add_argument('--cat_feat_gc', action='store_true')

    # loss
    parser.add_argument('--loss_fn', type=str, default='mse', choices=['mse', 'mae', 'mse_u', 'mae_u'])
    parser.add_argument('--lamda', type=float, default=2.0)

    # training
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--epochs', type=int, default=300, help='')
    parser.add_argument('--clip', type=int, default=3, help='Gradient Clipping')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.97, help='learning rate')
    parser.add_argument('--patience', type=int, default=20, help='quit if no improvement after this many iterations')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--plot', action='store_true')

    # parameter for test_routing
    parser.add_argument('--run_te', type=str, choices=['None', 'gwn_ls2sr'],
                        default='None')

    # rand: topk random
    # prand: topk + partially random in each routing cycle
    # train: topk from training set
    # gt: exact topk in each routing cycle
    # pred: topk from the predicted reconstructed TM
    parser.add_argument('--fs', help='Flow selection strategiy', type=str, choices=['rand',
                                                                                    'prand',
                                                                                    'train',
                                                                                    'gt',
                                                                                    'pred'],
                        default='train')

    parser.add_argument('--timeout', type=float, default=60.0)

    parser.add_argument('--te_step', type=int, default=0)
    parser.add_argument('--nrun', type=int, default=30)

    # get args
    args = parser.parse_args()

    if args.type == 'p1':
        args.out_seq_len = args.seq_len_y
    elif args.type == 'p2':
        args.out_seq_len = 1
    elif args.type == 'p3':
        if args.seq_len_y % args.trunk != 0:
            args.seq_len_y = int(args.seq_len_y / args.trunk) * args.trunk
        args.out_seq_len = args.trunk

    if 'geant' in args.dataset:
        args.intv = 15
    elif 'abilene' in args.dataset:
        args.intv = 5
    elif 'brain' in args.dataset:
        args.intv = 1
    else:
        args.intv = 5

    if 'geant' in args.dataset or 'renater' in args.dataset:
        args.seq_len_y = args.seq_len_x
        if args.seq_len_y == 6:
            args.blocks = 2
            args.layers = 2
            args.kernel_size = 2
            args.stride = 2
        elif args.seq_len_y == 12:
            args.blocks = 4
            args.layers = 2
            args.kernel_size = 2
            args.stride = 2
        elif args.seq_len_y == 18:
            args.blocks = 3
            args.layers = 2
            args.kernel_size = 3
            args.stride = 3
        elif args.seq_len_y == 24:
            args.blocks = 4
            args.layers = 2
            args.kernel_size = 3
            args.stride = 3
        elif args.seq_len_y == 36:
            args.blocks = 3
            args.layers = 1
            args.kernel_size = 4
            args.stride = 4
        elif args.seq_len_y == 48:
            args.blocks = 4
            args.layers = 2
            args.kernel_size = 4
            args.stride = 4
        elif args.seq_len_y == 60:
            args.blocks = 6
            args.layers = 2
            args.kernel_size = 4
            args.stride = 4
        elif args.seq_len_y == 72:
            args.blocks = 5
            args.layers = 2
            args.kernel_size = 4
            args.stride = 4
    else:
        args.seq_len_y = args.seq_len_x
        if args.seq_len_y == 6:
            args.blocks = 2
            args.layers = 2
            args.kernel_size = 2
            args.stride = 2
        elif args.seq_len_y == 12:
            args.blocks = 4
            args.layers = 2
            args.kernel_size = 2
            args.stride = 2
        elif args.seq_len_y == 24:
            args.blocks = 4
            args.layers = 3
            args.kernel_size = 2
            args.stride = 2
        elif args.seq_len_y == 36:
            args.blocks = 5
            args.layers = 3
            args.kernel_size = 2
            args.stride = 2
        elif args.seq_len_y == 48:
            args.blocks = 5
            args.layers = 2
            args.kernel_size = 4
            args.stride = 4
        elif args.seq_len_y == 60:
            args.blocks = 5
            args.layers = 1
            args.kernel_size = 4
            args.stride = 4
        elif args.seq_len_y == 72:
            args.blocks = 5
            args.layers = 2
            args.kernel_size = 4
            args.stride = 4

    return args


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
    print('    - granularity scale      :', args.k)
    print('    - num_series             :', args.nSeries)
    print('    - random measure rate    : {}%'.format(args.mon_rate))
    print('    - log path               :', args.log_dir)
    print('---------------------------------------------------------')
    print('    - model                  :', args.model)
    print('    - wn_blocks              :', args.blocks)
    print('    - wn_layers              :', args.layers)
    print('    - hidden                 :', args.hidden)
    print('    - kernel_size            :', args.kernel_size)
    print('    - stride                 :', args.stride)
    print('    - dilation_channels      :', args.dilation_channels)
    print('    - residual_channels      :', args.residual_channels)
    print('    - end_channels           :', args.end_channels)
    print('    - skip_channels          :', args.skip_channels)
    print('----------------------------')
    print('    - do_graph_conv          :', args.do_graph_conv)
    print('    - adjtype                :', args.adjtype)
    print('    - aptonly                :', args.aptonly)
    print('    - adjdata                :', args.adjdata)
    print('    - addaptadj              :', args.addaptadj)
    print('    - randomadj              :', args.randomadj)
    print('----------------------------')
    print('    - type                   :', args.type)
    print('    - seq_len_x              :', args.seq_len_x)
    print('    - seq_len_y              :', args.seq_len_y)
    print('    - out_seq_len            :', args.out_seq_len)
    print('    - tod                    :', args.tod)
    print('    - ma                     :', args.ma)
    print('    - mx                     :', args.mx)
    print('---------------------------------------------------------')
    print('    - device                 :', args.device)
    print('    - train_batch_size       :', args.train_batch_size)
    print('    - val_batch_size         :', args.val_batch_size)
    print('    - test_batch_size        :', args.test_batch_size)
    print('    - epochs                 :', args.epochs)
    print('    - learning_rate          :', args.learning_rate)
    print('    - patience               :', args.patience)
    print('    - plot_results           :', args.plot)
    print('---------------------------------------------------------')
    print('    - run te                 :', args.run_te)
    # print('    - routing        :', args.routing)
    # print('    - mon_policy     :', args.mon_policy)
    print('    - te_step                :', args.te_step)
    print('---------------------------------------------------------')
