import argparse
import os
import subprocess as sp

import numpy as np
from tqdm import trange

MON_RATE = np.arange(1, 51)


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
    parser.add_argument('--run_te', type=str, choices=['None', 'gwn_ls2sr'],
                        default='None')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--fs', help='Flow selection strategy', type=str, choices=['train'],
                        default='train')
    parser.add_argument('--mon_rate', type=int, default=0, choices=MON_RATE)
    parser.add_argument('--cs', type=int, default=-1, choices=[-1, 0, 1])
    parser.add_argument('--nrun', type=int, default=30)
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--epochs', type=int, default=300, help='')
    parser.add_argument('--seq_len_x', type=int, default=12, choices=[6, 12, 18, 24, 30, 36, 48],
                        help='input length default 64')

    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()
    dataset_name = args.dataset
    if args.mon_rate == 0:
        mon_rate = MON_RATE
    else:
        mon_rate = [args.mon_rate]

    if args.test:
        if args.cs == -1:
            CS = [0, 1]
        else:
            CS = [args.cs]
    else:
        CS = [1]
    iteration = trange(len(mon_rate))
    # experiment for each dataset
    for d in iteration:
        for cs in CS:
            cmd = 'python train_sappo.py --do_graph_conv --aptonly --addaptadj --randomadj'
            cmd += ' --train_batch_size 64 --val_batch_size 64'
            cmd += ' --dataset {}'.format(dataset_name)
            cmd += ' --mon_rate {}'.format(mon_rate[d])
            cmd += ' --device {}'.format(args.device)
            cmd += ' --fs {}'.format(args.fs)
            cmd += ' --epochs {}'.format(args.epochs)
            cmd += ' --seq_len_x {}'.format(args.seq_len_x)

            if args.test:
                cmd += ' --test'
                cmd += ' --cs {}'.format(cs)

            if args.run_te != 'None':
                cmd += ' --run_te {}'.format(args.run_te)
                cmd += ' --nrun {}'.format(args.nrun)
                cmd += ' --timeout {}'.format(args.timeout)

            print(cmd)
            os.system(cmd)
            iteration.set_description(
                'Dataset {} mon_rate: {} - cs {}'.format(dataset_name, mon_rate[d], cs))


if __name__ == '__main__':
    main()
