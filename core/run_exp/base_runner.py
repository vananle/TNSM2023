import sys

sys.path.append('../../../')

import time
import warnings

from core import prediction_models

from core import training
from core import routing
from datetime import date
from core import utils
from core.utils import *
import torch

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


def run_exp(args):
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

    metrics = {}
    if args.model != 'vae':
        metrics = engine.test()

        if 'cs' in args.method:
            engine.tm_reconstruction()
            te_data = {'test/y_hat': engine.data['test/y_hat'],
                       'test/x_gt': engine.data['test/x_gt'],
                       'test/y_gt': engine.data['test/y_gt'],
                       'test/y_cs': engine.data['test/y_cs'],
                       'mae_y_cs': engine.data['mae_y_cs']
                       }
            metrics.update({'mae_y_cs': engine.data['mae_y_cs']})
        else:

            te_data = {'test/y_hat': engine.data['test/y_hat'],
                       'test/x_gt': engine.data['test/x_gt'],
                       'test/y_gt': engine.data['test/y_gt'],
                       'scaler': engine.data['scaler']
                       }
        engine.save_data(te_data)
    else:
        te_data = {'test/x_gt': engine.data['test/x_gt'],
                   'test/y_gt': engine.data['test/y_gt'],
                   'scaler': engine.data['scaler']
                   }

    # args.monitor = engine.monitor
    # mlu, rc = 0, 0
    # if args.te_alg != 'None':
    #     te = routing.TrafficEngineering(args, data=te_data)
    #     if args.model == 'vae':
    #         mlu, rc = te.vae_ls2sr(engine.model)
    #     else:
    #         mlu, rc = te.run_te()

    return metrics, mlu, rc


def run_te(args):
    device = torch.device(args.device)
    args.device = device

    monitor = Monitor(args, args.num_epochs)

    print('[+] Logs:', monitor.label)

    te_data = monitor.load_data()

    args.monitor = monitor
    mlu, rc = 0, 0
    if args.te_alg != 'None':
        te = routing.TrafficEngineering(args, data=te_data)
        mlu, rc = te.run_te()

    return metrics, mlu, rc


def exp_11():
    args = utils.get_args()
    t1 = time.time()
    input_len = 15
    datasets = ['abilene', 'geant']
    model = 'gwn'
    te_algs = ['p1', 'p2', 'p3']
    predict_len = [6]
    seed = 20

    for dataset_id, dataset in enumerate(datasets):
        results = {'mlu': np.zeros(shape=(len(te_algs), len(predict_len))),
                   'rc': np.zeros(shape=(len(te_algs), len(predict_len)))}

        for te_alg_id, te_alg in enumerate(te_algs):
            for pred_len_id, pre_len in enumerate(predict_len):
                args.input_len = input_len
                args.predict_len = pre_len
                args.dataset = dataset
                args.model = model
                args.seed = seed

                args.train_batch_size = 32
                args.val_batch_size = 32

                args.test = True
                args.te_alg = te_alg
                args.timeout = 1

                args = utils.args_adjust(args)

                _, mlu, rc = run_te(args)
                results['mlu'][te_alg_id, pred_len_id] = np.mean(mlu)
                results['rc'][te_alg_id, pred_len_id] = np.mean(rc)

        os.makedirs('../results/core/exp11/', exist_ok=True)
        for k, v in results.items():
            np.savetxt(f'../results/core/exp11/exp11_{dataset}_{k}.txt', results[k], delimiter=',')

    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())


def exp_12():
    args = utils.get_args()
    t1 = time.time()
    input_len = 15
    datasets = ['abilene', 'geant']
    model = 'gwn'
    # te_algs = ['p0', 'ob']
    te_algs = ['sp']
    predict_len = [6]

    for dataset_id, dataset in enumerate(datasets):
        results = {'mlu': np.zeros(shape=(len(te_algs), len(predict_len))),
                   'rc': np.zeros(shape=(len(te_algs), len(predict_len)))}

        for te_alg_id, te_alg in enumerate(te_algs):
            for pred_len_id, pre_len in enumerate(predict_len):
                args.input_len = input_len
                args.predict_len = pre_len
                args.dataset = dataset
                args.model = model

                args.train_batch_size = 32
                args.val_batch_size = 32

                args.test = True
                args.te_alg = te_alg
                args.timeout = 1
                args = utils.args_adjust(args)

                _, mlu, rc = run_exp(args)
                results['mlu'][te_alg_id, pred_len_id] = np.mean(mlu)
                results['rc'][te_alg_id, pred_len_id] = np.mean(rc)

        os.makedirs('../results/core/exp12/', exist_ok=True)
        for k, v in results.items():
            np.savetxt(f'../results/core/exp12/exp_12{dataset}_{k}.txt', results[k], delimiter=',')

    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())


def exp_6():
    args = utils.get_args()
    t1 = time.time()
    input_len = 12
    datasets = ['abilene', 'geant', 'germany']
    models = ['gwn', 'lstm', 'gru']
    topkflows = [0.6, 0.7, 0.8, 0.9]
    pre_len = 6
    method = 'mtsr_cs'

    for dataset_id, dataset in enumerate(datasets):
        results = {'loss': np.zeros(shape=(len(models), len(topkflows))),
                   'rse': np.zeros(shape=(len(models), len(topkflows))),
                   'mae': np.zeros(shape=(len(models), len(topkflows))),
                   'mse': np.zeros(shape=(len(models), len(topkflows))),
                   'mape': np.zeros(shape=(len(models), len(topkflows))),
                   'rmse': np.zeros(shape=(len(models), len(topkflows))),
                   'mlu': np.zeros(shape=(len(models), len(topkflows))),
                   'rc': np.zeros(shape=(len(models), len(topkflows)))}

        for model_id, model in enumerate(models):
            for topk_id, topk in enumerate(topkflows):
                args.input_len = input_len
                args.predict_len = pre_len
                args.dataset = dataset
                args.model = model
                args.topk = topk
                args.method = method

                if args.predict_len >= 12 or args.dataset == 'germany':
                    args.train_batch_size = 64
                    args.val_batch_size = 64
                else:
                    args.train_batch_size = 128
                    args.val_batch_size = 128

                args = utils.args_adjust(args)
                args.test = False
                args.te_alg = 'srls'
                args.timeout = 1
                metrics, mlu, rc = run_exp(args)
                for k, v in metrics.items():
                    results[k][model_id, topk_id] = v
                results['mlu'][model_id, topk_id] = np.mean(mlu)
                results['rc'][model_id, topk_id] = np.mean(rc)
                print(dataset, model, pre_len, metrics)

        os.makedirs('../results/core/', exist_ok=True)
        for k, v in results.items():
            np.savetxt(f'../results/core/exp6.2_{dataset}_{k}.txt', results[k], delimiter=',')

    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())
