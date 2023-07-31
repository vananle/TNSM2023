import itertools
import sys

sys.path.append('../../')

import time
import warnings

import prediction_models
import utils

import training
import routing
from datetime import date
from joblib import delayed, Parallel
from core.utils import *

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

    args.monitor = engine.monitor
    mlu, rc = 0, 0
    if args.te_alg != 'None':
        te = routing.TrafficEngineering(args, data=te_data)
        if args.model == 'vae':
            mlu, rc = te.vae_ls2sr(engine.model)
        else:
            mlu, rc = te.run_te()

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


def exp_1():
    print('|----- RUN EXP 1: TRAFFIC PREDICTION WITH DIFFERENT PRED_LEN ----')
    args = utils.get_args()
    t1 = time.time()
    input_len = 15
    # datasets = ['abilene', 'geant']
    datasets = ['germany', 'gnnet-40']
    models = ['gwn', 'lstm', 'gru', 'stgcn', 'mtgnn']
    predict_len = [3, 6, 9, 12]
    seeds = [20, 5, 1, 46, 77]
    # seeds are randomly generated from seeds = np.random.choice(np.arange(100), size=10, replace=False)
    for seed in seeds:
        for dataset_id, dataset in enumerate(datasets):
            results = {'loss': np.zeros(shape=(len(models), len(predict_len))),
                       'rse': np.zeros(shape=(len(models), len(predict_len))),
                       'mae': np.zeros(shape=(len(models), len(predict_len))),
                       'mse': np.zeros(shape=(len(models), len(predict_len))),
                       'mape': np.zeros(shape=(len(models), len(predict_len))),
                       'rmse': np.zeros(shape=(len(models), len(predict_len))),
                       'mlu': np.zeros(shape=(len(models), len(predict_len))),
                       'rc': np.zeros(shape=(len(models), len(predict_len))),
                       }

            for model_id, model in enumerate(models):
                for pred_len_id, pre_len in enumerate(predict_len):
                    args.input_len = input_len
                    args.predict_len = pre_len
                    args.dataset = dataset
                    args.model = model

                    args.train_batch_size = 32
                    args.val_batch_size = 32
                    args.seed = seed

                    args = utils.args_adjust(args)
                    args.test = False
                    args.te_alg = 'srls'
                    args.timeout = 1
                    metrics, mlu, rc = run_exp(args)
                    for k, v in metrics.items():
                        results[k][model_id, pred_len_id] = v

                    results['mlu'][model_id, pred_len_id] = np.mean(mlu)
                    results['rc'][model_id, pred_len_id] = np.mean(rc)

                    print(dataset, model, pre_len, metrics)

            os.makedirs('../results/core/exp1', exist_ok=True)
            for k, v in results.items():
                np.savetxt(f'../results/core/exp1/mtsr_prediction_error_vs_prediction_len_{dataset}_{k}_{seed}.txt',
                           results[k], delimiter=',')

    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())


def exp_2():
    args = utils.get_args()
    t1 = time.time()
    input_len = [12, 15, 18, 21]
    datasets = ['germany', 'gnnet-40']
    # datasets = ['abilene', 'geant']
    models = ['gwn', 'lstm', 'gru', 'stgcn', 'mtgnn']
    pre_len = 6
    seeds = [20, 5, 1, 46, 77]

    for seed in seeds:
        for dataset_id, dataset in enumerate(datasets):
            results = {'loss': np.zeros(shape=(len(models), len(input_len))),
                       'rse': np.zeros(shape=(len(models), len(input_len))),
                       'mae': np.zeros(shape=(len(models), len(input_len))),
                       'mse': np.zeros(shape=(len(models), len(input_len))),
                       'mape': np.zeros(shape=(len(models), len(input_len))),
                       'rmse': np.zeros(shape=(len(models), len(input_len))),
                       'mlu': np.zeros(shape=(len(models), len(input_len))),
                       'rc': np.zeros(shape=(len(models), len(input_len)))
                       }

            for model_id, model in enumerate(models):
                for in_len_id, in_len in enumerate(input_len):

                    # if in_len == 15:
                    #     continue

                    args.input_len = in_len
                    args.predict_len = pre_len
                    args.dataset = dataset
                    args.model = model

                    args.train_batch_size = 32
                    args.val_batch_size = 32

                    args.seed = seed

                    args = utils.args_adjust(args)
                    args.test = True
                    args.te_alg = 'srls'
                    args.timeout = 1
                    metrics, mlu, rc = run_exp(args)
                    for k, v in metrics.items():
                        results[k][model_id, in_len_id] = v
                    results['mlu'][model_id, in_len_id] = np.mean(mlu)
                    results['rc'][model_id, in_len_id] = np.mean(rc)

                    print(dataset, model, pre_len, metrics)

            os.makedirs('../results/core/exp2/', exist_ok=True)
            for k, v in results.items():
                np.savetxt(f'../results/core/exp2/mtsr_prediction_error_vs_input_len_{dataset}_{k}_{seed}.txt',
                           results[k], delimiter=',')

    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())


def exp_3():
    print('RUN EXP 3')
    args = utils.get_args()
    t1 = time.time()
    input_len = 15
    datasets = ['germany', 'gnnet-40']
    # datasets = ['abilene', 'geant']
    models = ['gwn', 'lstm', 'gru', 'stgcn', 'mtgnn']
    predict_len = [3, 6, 9, 12, 15]
    seeds = [20, 5, 1, 46, 77]

    def run_exp_parallel(seed, dataset, model, pre_len, args):
        args.input_len = input_len
        args.predict_len = pre_len
        args.dataset = dataset
        args.model = model
        args.seed = seed

        args.train_batch_size = 32
        args.val_batch_size = 32

        device = np.random.randint(0, 1)
        args.device = f'cuda:{device}'

        args = utils.args_adjust(args)
        args.test = True
        args.te_alg = 'srls'
        args.timeout = 1
        metrics, mlu, rc = run_te(args)

        return mlu, rc

    ret = Parallel(n_jobs=16)(delayed(run_exp_parallel)(seed, dataset, model, pre_len, args)
                              for seed, dataset, model, pre_len in itertools.product(seeds, datasets,
                                                                                     models, predict_len))
    i = 0
    for seed in seeds:
        for dataset_id, dataset in enumerate(datasets):
            results = {'mlu': np.zeros(shape=(len(models), len(predict_len))),
                       'rc': np.zeros(shape=(len(models), len(predict_len)))}

            for model_id, model in enumerate(models):
                for pred_len_id, pre_len in enumerate(predict_len):
                    mlu, rc = ret[i]
                    results['mlu'][model_id, pred_len_id] = np.mean(mlu)
                    results['rc'][model_id, pred_len_id] = np.mean(rc)

                    i += 1

            os.makedirs('../results/core/exp3/', exist_ok=True)
            for k, v in results.items():
                np.savetxt(f'../results/core/exp3/exp3_{dataset}_{k}_{seed}.txt', results[k], delimiter=',')

    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())


def exp_4():
    args = utils.get_args()
    t1 = time.time()
    input_lens = [12, 15, 18, 21]
    datasets = ['abilene', 'geant']
    models = ['gwn', 'lstm', 'gru', 'stgcn', 'mtgnn']
    pre_len = 6
    seeds = [20, 5, 1, 46, 77]

    def run_exp_parallel(seed, dataset, model, input_len, args):
        args.input_len = input_len
        args.predict_len = pre_len
        args.dataset = dataset
        args.model = model
        args.seed = seed

        args.train_batch_size = 32
        args.val_batch_size = 32

        device = np.random.randint(0, 1)
        args.device = f'cuda:{device}'

        args = utils.args_adjust(args)
        args.test = True
        args.te_alg = 'srls'
        args.timeout = 1
        metrics, mlu, rc = run_te(args)

        return mlu, rc

    ret = Parallel(n_jobs=8)(delayed(run_exp_parallel)(seed, dataset, model, input_len, args)
                             for seed, dataset, model, input_len in itertools.product(seeds, datasets,
                                                                                      models, input_lens))
    i = 0
    for seed in seeds:

        for dataset_id, dataset in enumerate(datasets):
            results = {'mlu': np.zeros(shape=(len(models), len(input_lens))),
                       'rc': np.zeros(shape=(len(models), len(input_lens)))}

            for model_id, model in enumerate(models):
                for input_len_id, pre_len in enumerate(input_lens):
                    mlu, rc = ret[i]
                    results['mlu'][model_id, input_len_id] = np.mean(mlu)
                    results['rc'][model_id, input_len_id] = np.mean(rc)

                    i += 1

            os.makedirs('../results/core/exp4/', exist_ok=True)
            for k, v in results.items():
                np.savetxt(f'../results/core/exp4/exp4_{dataset}_{k}_{seed}.txt', results[k], delimiter=',')

    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())


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


def exp_8():
    print('RUNNNING EXP8')

    args = utils.get_args()
    t1 = time.time()
    input_len = 15
    datasets = ['germany', 'gnnet-40']
    # datasets = ['abilene', 'geant']
    models = ['gwn']
    mon_per = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pre_len = 6
    method = 'mtsr_cs'
    mon_method = 'random'

    seeds = [20, 5, 1, 46, 77]

    for seed in seeds:
        for dataset_id, dataset in enumerate(datasets):
            results = {'loss': np.zeros(shape=(len(models), len(mon_per))),
                       'rse': np.zeros(shape=(len(models), len(mon_per))),
                       'mae': np.zeros(shape=(len(models), len(mon_per))),
                       'mse': np.zeros(shape=(len(models), len(mon_per))),
                       'mape': np.zeros(shape=(len(models), len(mon_per))),
                       'rmse': np.zeros(shape=(len(models), len(mon_per))),
                       'mae_y_cs': np.zeros(shape=(len(models), len(mon_per))),
                       'mlu': np.zeros(shape=(len(models), len(mon_per))),
                       'rc': np.zeros(shape=(len(models), len(mon_per)))}

            for model_id, model in enumerate(models):
                for mon_p_id, mon_p in enumerate(mon_per):
                    args.input_len = input_len
                    args.predict_len = pre_len
                    args.dataset = dataset
                    args.model = model
                    args.mon_method = mon_method
                    args.mon_per = mon_p
                    args.method = method
                    args.seed = seed

                    args.train_batch_size = 32
                    args.val_batch_size = 32

                    args = utils.args_adjust(args)
                    args.test = False
                    args.te_alg = 'srls'
                    args.timeout = 1
                    metrics, mlu, rc = run_exp(args)
                    for k, v in metrics.items():
                        results[k][model_id, mon_p_id] = v
                    results['mlu'][model_id, mon_p_id] = np.mean(mlu)
                    results['rc'][model_id, mon_p_id] = np.mean(rc)
                    print(dataset, model, pre_len, metrics)

            os.makedirs('../results/core/exp8', exist_ok=True)
            for k, v in results.items():
                np.savetxt(f'../results/core/exp8/mtsr_cs_{method}_{mon_method}_{dataset}_{k}_{seed}.txt', results[k],
                           delimiter=',')

    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())


def exp_7():
    print('RUNNNING EXP 7')

    args = utils.get_args()
    t1 = time.time()
    input_len = 15
    datasets = ['abilene', 'geant']
    models = ['gwn']
    mon_per = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pre_len = 6
    method = 'mtsr_nocs'
    mon_method = 'random'

    seeds = [20, 5, 1, 46, 77]

    for seed in seeds:

        for dataset_id, dataset in enumerate(datasets):
            results = {
                'mlu': np.zeros(shape=(len(models), len(mon_per))),
                'rc': np.zeros(shape=(len(models), len(mon_per)))}

            for model_id, model in enumerate(models):
                for mon_p_id, mon_p in enumerate(mon_per):
                    args.input_len = input_len
                    args.predict_len = pre_len
                    args.dataset = dataset
                    args.model = model
                    args.mon_method = mon_method
                    args.mon_per = mon_p
                    args.method = method
                    args.seed = seed

                    args.train_batch_size = 32
                    args.val_batch_size = 32

                    args = utils.args_adjust(args)
                    args.test = True
                    args.te_alg = 'srls'
                    args.timeout = 1
                    _, mlu, rc = run_exp(args)

                    results['mlu'][model_id, mon_p_id] = np.mean(mlu)
                    results['rc'][model_id, mon_p_id] = np.mean(rc)
                    print(dataset, model, pre_len, metrics)

            os.makedirs('../results/core/exp7', exist_ok=True)
            for k, v in results.items():
                np.savetxt(f'../results/core/exp7/mtsr_nocs_{method}_{mon_method}_{dataset}_{k}_{seed}.txt', results[k],
                           delimiter=',')

    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())


def exp_9():
    args = utils.get_args()
    t1 = time.time()
    input_len = [12]
    datasets = ['abilene', 'geant']
    model = 'gwn'
    te_algs = ['p1', 'p2']
    predict_len = [6]
    method = 'core'

    for dataset_id, dataset in enumerate(datasets):
        results = {'mlu': np.zeros(shape=(len(te_algs), len(predict_len))),
                   'rc': np.zeros(shape=(len(te_algs), len(predict_len)))}

        for te_alg_id, te_alg in enumerate(te_algs):
            for pred_len_id, pre_len in enumerate(predict_len):
                args.input_len = 12
                args.predict_len = pre_len
                args.dataset = dataset
                args.model = model
                args.method = method

                if args.predict_len >= 12 or args.dataset == 'germany':
                    args.train_batch_size = 64
                    args.val_batch_size = 64
                else:
                    args.train_batch_size = 128
                    args.val_batch_size = 128

                args = utils.args_adjust(args)
                args.test = True
                args.te_alg = te_alg
                args.timeout = 60
                _, mlu, rc = run_te(args)
                results['mlu'][te_alg_id, pred_len_id] = np.mean(mlu)
                results['rc'][te_alg_id, pred_len_id] = np.mean(rc)

        os.makedirs('../results/core/exp9/', exist_ok=True)
        for k, v in results.items():
            np.savetxt(f'../results/core/exp9/exp9_{dataset}_{k}.txt', results[k], delimiter=',')

    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())


def exp_10():
    print('RUNNNING EXP10 VAE')

    args = utils.get_args()
    t1 = time.time()
    input_len = 15
    datasets = ['abilene', 'geant']
    models = ['vae']
    pre_len = 6
    method = 'mtsr_cs'
    mon_method = 'random'

    seeds = [20, 5, 1, 46, 77]
    # seeds = [77]

    for seed in seeds:

        for dataset_id, dataset in enumerate(datasets):
            results = {'mlu': [],
                       'rc': []}

            for model_id, model in enumerate(models):
                args.input_len = input_len
                args.predict_len = pre_len
                args.dataset = dataset
                args.model = model
                args.mon_method = mon_method
                args.method = method
                args.seed = seed

                args.train_batch_size = 32
                args.val_batch_size = 32

                args = utils.args_adjust(args)
                args.test = True
                args.te_alg = 'srls'
                args.timeout = 1

                _, mlu, rc = run_exp(args)
                results['mlu'].append(np.mean(mlu))
                results['rc'].append(np.mean(mlu))

            os.makedirs('../results/core/exp10/', exist_ok=True)
            for k, v in results.items():
                np.savetxt(f'../results/core/exp10/mtsr_cs_vae_{dataset}_{k}_{seed}.txt', results[k], delimiter=',')

    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())


# exp_1()
# exp_2()
# exp_3()
# exp_4()
# exp_5()
# exp_6()
# exp_7()
exp_8()
# exp_12()
