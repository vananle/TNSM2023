import argparse

ADJ_CHOICES = ['scalap', 'normlap', 'symnadj', 'transition', 'identity']


def get_args():
    # create argument parser
    parser = argparse.ArgumentParser()

    # parameter for dataset
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='abilene_tm',
                        choices=['abilene_tm', 'geant_tm', 'brain_tm', 'renater_tm', 'surfnet_tm', 'uninett_tm',
                                 'geant3_tm'],
                        help='Dataset, (default abilene_tm)')
    parser.add_argument('--datapath', type=str, default='thesis_data/dataset')
    parser.add_argument('--type', type=str, default='p2', choices=['p1', 'p2', 'p3'],
                        help='problem formulation (default p2)')
    parser.add_argument('--trunk', type=int, default=3, help='trunk for p3 problem (default 3)')
    parser.add_argument('--k', type=int, default=1, help='granularity scale', choices=[1, 2, 3])

    parser.add_argument('--tod', action='store_true')
    parser.add_argument('--ma', action='store_true')
    parser.add_argument('--mx', action='store_true')

    # Model
    # Graph
    parser.add_argument('--model', type=str, default='vae')

    # Wavenet
    parser.add_argument('--seq_len_x', type=int, default=12, choices=[6, 12, 18, 24, 30],
                        help='input length default 12')
    parser.add_argument('--seq_len_y', type=int, default=12, choices=[6, 12, 18, 24, 30],
                        help='routing cycle 12')

    parser.add_argument('--latent_dim', type=int, default=10, help='latent_dim')
    parser.add_argument('--scale', type=int, default=100, help='scale')

    # loss
    parser.add_argument('--loss_fn', type=str, default='mae', choices=['mse', 'mae', 'mse_u', 'mae_u'])
    parser.add_argument('--lamda', type=float, default=2.0)

    # training
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--epochs', type=int, default=1000, help='')
    parser.add_argument('--clip', type=int, default=3, help='Gradient Clipping')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='learning rate')
    parser.add_argument('--patience', type=int, default=50, help='quit if no improvement after this many iterations')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--plot', action='store_true')

    # parameter for test_routing
    parser.add_argument('--ncf', default=10, type=int, help='default 10')

    parser.add_argument('--timeout', type=float, default=1.0)
    parser.add_argument('--te_step', type=int, default=0)
    parser.add_argument('--nrun', type=int, default=3)

    # get args
    args = parser.parse_args()

    if args.type == 'p1':
        args.output_len = args.seq_len_y
    elif args.type == 'p2':
        args.output_len = 1
    elif args.type == 'p3':
        if args.seq_len_y % args.trunk != 0:
            args.seq_len_y = int(args.seq_len_y / args.trunk) * args.trunk
        args.output_len = args.trunk

    args.seq_len_y = args.seq_len_x

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
    print('    - adjdata_path           :', args.adjdata_path)
    print('    - addaptadj              :', args.addaptadj)
    print('    - randomadj              :', args.randomadj)
    print('----------------------------')
    print('    - type                   :', args.type)
    print('    - seq_len_x              :', args.seq_len_x)
    print('    - seq_len_y              :', args.seq_len_y)
    print('    - output_len            :', args.output_len)
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
    print('    - te_step                :', args.te_step)
    print('    - ncf                    :', args.ncf)
    print('---------------------------------------------------------')
