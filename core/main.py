import sys

import numpy as np

sys.path.append('../../')

import time
import warnings

import torch

import prediction_models
import utils

import training
import routing
from datetime import date

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


def main(args):
    device = torch.device(args.device)
    args.device = device

    model = prediction_models.create_model(args)

    if args.model == 'vae':
        data = utils.get_data_vae(args)
        engine = training.TrainEngine_VAE(data=data, model=model, args=args)
    else:
        data = utils.get_dataloader(args)
        engine = training.TrainEngine(data=data, model=model, args=args)

    # utils.print_args(args)
    print('[+] Logs:', engine.monitor.label)

    if not args.test:
        engine.train()

    else:
        if args.model == 'vae':
            pass
        else:
            metrics = engine.test()
            if 'cs' in args.method:
                engine.tm_reconstruction()

            print(metrics)

        args.monitor = engine.monitor
        if args.te_alg != 'None':
            te = routing.TrafficEngineering(args, data=engine.data)
            if args.model == 'vae':
                mlu, rc = te.vae_ls2sr(engine.model)
            else:
                mlu, rc = te.run_te()
            print('MLU: {}'.format(np.mean(mlu)))
            print('RC: {}'.format(np.mean(rc)))


if __name__ == "__main__":
    args = utils.get_args()
    args = utils.args_adjust(args)
    if args.predict_len >= 12 or args.dataset == 'germany' or args.input_len > 12:
        args.train_batch_size = 32
        args.val_batch_size = 32
    else:
        args.train_batch_size = 128
        args.val_batch_size = 128

    t1 = time.time()
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())
