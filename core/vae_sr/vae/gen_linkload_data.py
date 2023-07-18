import sys

sys.path.append('../../../')

import os.path
import time
import warnings
from datetime import date

import torch

from core.vae_sr.vae import utils
from core.vae_sr.vae.utils.load_data import load_data
from core.routing.do_te import vae_gen_data, createGraph_srls

# sys.path.append('..')

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


def main(args, **model_kwargs):
    homedir = os.path.expanduser('~')
    args.datapath = os.path.join(homedir, args.datapath)

    device = torch.device(args.device)
    args.device = device
    if 'abilene' in args.dataset:
        args.nNodes = 12
        args.nLinks = 30
        args.day_size = 288
        scale = 100.0
    elif 'geant' in args.dataset:
        args.nNodes = 22
        args.nLinks = 72
        args.day_size = 96
        scale = 100.0
    elif 'brain' in args.dataset:
        args.nNodes = 9
        args.nLinks = 28
        args.day_size = 1440
    else:
        raise ValueError('Dataset not found!')
    logger = utils.Logger(args)
    tm_train, tm_val, tm_test, n_node = load_data(args=args)

    nx_graph = createGraph_srls(os.path.join(args.datapath, 'topo/{}_node.csv'.format(args.dataset)),
                                os.path.join(args.datapath, 'topo/{}_edge.csv'.format(args.dataset)))

    vae_gen_data(data=tm_train, graphs=nx_graph, args=args, fname='train')
    # vae_gen_data(dataset=tm_val, graphs=nx_graph, args=args, fname='val')


if __name__ == "__main__":
    args = utils.get_args()
    t1 = time.time()
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())
