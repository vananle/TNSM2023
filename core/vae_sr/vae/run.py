import argparse
import os
import subprocess as sp


def call(args):
    p = sp.run(args=args,
               stdout=sp.PIPE,
               stderr=sp.PIPE)
    stdout = p.stdout.decode('utf-8')
    return stdout


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='abilene_tm',
                        choices=['abilene_tm', 'geant_tm', 'brain_tm', 'uninett_tm', 'renater_tm', 'geant3_tm'],
                        help='Dataset, (default abilene_tm)')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--testset', type=int, default=-1,
                        choices=[-1, 0, 1, 2, 3, 4],
                        help='Test set, (default -1 run all test)')
    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--timeout', type=float, default=1.0)
    parser.add_argument('--nrun', type=int, default=3)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seq_len_x', type=int, default=12, choices=[6, 12, 18, 24, 30],
                        help='input length default 64')
    parser.add_argument('--seq_len_y', type=int, default=12, choices=[6, 12, 18, 24, 30],
                        help='routing cycle 12')
    parser.add_argument('--ncf', default=10, type=int, help='default 10')

    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()
    dataset_name = args.dataset
    if args.run_te == 'all':
        run_te = ['gwn_ls2sr', 'gt_ls2sr', 'p0', 'p1', 'p2', 'gwn_p2', 'p3', 'onestep',
                  'laststep', 'laststep_ls2sr', 'firststep', 'or']
    else:
        run_te = [args.run_te]
    # run_te = ['None', 'gwn_ls2sr', 'gt_ls2sr', 'p0', 'p1', 'p2', 'gwn_p2', 'p3', 'onestep',
    #           'prophet', 'laststep', 'laststep_ls2sr', 'firststep', 'or']

    # experiment for each dataset
    cmd = 'python train_sappo.py --do_graph_conv --addaptadj --randomadj'
    cmd += ' --train_batch_size 64 --val_batch_size 64'
    cmd += ' --dataset {}'.format(dataset_name)
    cmd += ' --device {}'.format(args.device)
    cmd += ' --epochs {}'.format(args.epochs)
    cmd += ' --seq_len_x {}'.format(args.seq_len_x)
    cmd += ' --ncf {}'.format(args.ncf)

    if args.verbose:
        cmd += ' --verbose'

    if args.test:
        cmd += ' --test'
        if run_te[0] != 'None':
            for te in run_te:
                cmd += ' --timeout {}'.format(args.timeout)
                cmd += ' --nrun {}'.format(args.nrun)
                print(cmd)
                os.system(cmd)
        else:
            print(cmd)
            os.system(cmd)
    else:
        print(cmd)
        os.system(cmd)


if __name__ == '__main__':
    main()
