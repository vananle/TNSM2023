import sys

sys.path.append('../../../')
import os.path
import time
import warnings
from datetime import date

import tensorflow as tf
import torch

import models
from mtsr.vae_sr.vae import utils
from mtsr.routing.do_te import vae_ls2sr
from mtsr.routing.do_te import createGraph_srls

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
    X_train, X_val = utils.load_traindata(args)

    X_train = X_train / args.scale
    X_val = X_val / args.scale

    model = models.VAE(args=args, X_val=X_val)

    model.compile(optimizer=tf.keras.optimizers.Adam())
    print(model.encoder.summary())
    print(model.decoder.summary())
    if not args.test:
        history = model.fit(X_train, epochs=args.epochs, batch_size=args.train_batch_size,
                            callbacks=[utils.EarlyStoppingAtMinLoss()], )
        model.encoder.save(os.path.join(args.log_dir, 'encoder'))
        model.decoder.save(os.path.join(args.log_dir, 'decoder'))

    test_traffic = utils.load_test_traffic(args)
    nx_graph = createGraph_srls(os.path.join(args.datapath, 'topo/{}_node.csv'.format(args.dataset)),
                                os.path.join(args.datapath, 'topo/{}_edge.csv'.format(args.dataset)))

    graph, nNodes, nEdges, capacity, sp = nx_graph
    vae_ls2sr(test_traffic=test_traffic, vae=model, graph=graph, args=args, sp=sp, writer=logger.writer, fname='Test')


if __name__ == "__main__":
    args = utils.get_args()
    t1 = time.time()
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())
