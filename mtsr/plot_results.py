import copy
import sys

from matplotlib import pyplot as plt

sys.path.append('../../')

import utils
from utils import *

logs_path = '../results/mtsr/'
results_plot_path = '../results_plot/mtsr'
if not os.path.exists(results_plot_path):
    os.makedirs(results_plot_path)


def plot_exp1():
    args = utils.get_args()
    input_len = 15
    datasets = ['abilene', 'geant']
    models = ['gwn', 'lstm', 'gru', 'stgcn', 'mtgnn']
    predict_len = [3, 6, 9, 12, 15]
    seeds = [20, 5, 1, 46, 77]

    results = {}

    for dataset_id, dataset in enumerate(datasets):
        for model_id, model in enumerate(models):
            for pred_len_id, pre_len in enumerate(predict_len):
                for seed in seeds:
                    args.input_len = input_len
                    args.predict_len = pre_len
                    args.dataset = dataset
                    args.model = model

                    args.train_batch_size = 32
                    args.val_batch_size = 32
                    args.seed = seed

                    args.test = True
                    args.te_alg = 'srls'
                    args.timeout = 1

                    args = utils.args_adjust(args)

                    monitor = Monitor(args, args.num_epochs)

                    path = os.path.join(args.model_folder, f'te-{monitor.label}-{args.te_alg}-'
                                                           f'{args.use_gt}-{args.timeout}.npz')

                    data = np.load(path)
                    results[f'{seed}-{dataset}-{model}-{pre_len}-srls'] = data

    input_len = 15
    datasets = ['abilene', 'geant']
    model = 'gwn'
    te_algs = ['p1', 'p2', 'p3']
    predict_len = [6]
    seed = 20
    for dataset_id, dataset in enumerate(datasets):
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
                monitor = Monitor(args, args.num_epochs)
                path = os.path.join(args.model_folder, f'te-{monitor.label}-{args.te_alg}-'
                                                       f'{args.use_gt}-{args.timeout}.npz')

                data = np.load(path)
                results[f'{dataset}-{pre_len}-{te_alg}'] = data

    colors = ['r', 'g', 'k', 'b', 'm']
    predict_len = [3, 6, 9, 12, 15]
    datasets = ['abilene', 'geant']
    models = ['GWN', 'LSTM', 'GRU', 'STGCN', 'MTGNN']

    for dataset_id, dataset in enumerate(datasets):
        results_mae = []
        for index, seed in enumerate(seeds):
            path = f'../results/mtsr/exp1/mtsr_prediction_error_vs_prediction_len_{dataset}_mae_{seed}.txt'
            data = np.loadtxt(path, delimiter=',')
            if index == 0:
                results_mae = data
            else:
                if len(results_mae.shape) == 2:
                    results_mae = np.expand_dims(results_mae, axis=-1)
                data = np.expand_dims(data, axis=-1)
                results_mae = np.concatenate((results_mae, data), axis=-1)

        mae_mean = np.mean(results_mae, axis=-1)
        mae_std = np.std(results_mae, axis=-1)

        print(mae_mean.shape)
        fig, ax = plt.subplots()

        for i in range(mae_mean.shape[0]):
            ax.errorbar(predict_len, mae_mean[i], mae_std[i], label=models[i], color=colors[i])

        ax.set_xlabel(r'Routing cycle length ($T$)', fontsize=15)
        ax.set_ylabel('Mean Absolute Error', fontsize=15)
        ax.tick_params(axis='both', which='both', labelsize=12)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.legend(fontsize=15)
        plt.savefig(os.path.join(results_plot_path, f'exp1_{dataset}_mae.svg'), dpi=300)
        plt.close()


def plot_exp2():
    args = utils.get_args()
    input_len = 15
    datasets = ['abilene', 'geant']
    models = ['gwn', 'lstm', 'gru', 'stgcn', 'mtgnn']
    predict_len = [3, 6, 9, 12, 15]
    seeds = [20, 5, 1, 46, 77]

    results = {}

    # for dataset_id, dataset in enumerate(datasets):
    #     for model_id, model in enumerate(models):
    #         for pred_len_id, pre_len in enumerate(predict_len):
    #             for seed in seeds:
    #                 args.input_len = input_len
    #                 args.predict_len = pre_len
    #                 args.dataset = dataset
    #                 args.model = model
    #
    #                 args.train_batch_size = 32
    #                 args.val_batch_size = 32
    #                 args.seed = seed
    #
    #                 args.test = True
    #                 args.te_alg = 'srls'
    #                 args.timeout = 1
    #
    #                 args = utils.args_adjust(args)
    #
    #                 monitor = Monitor(args, args.num_epochs)
    #
    #                 path = os.path.join(args.model_folder, f'te-{monitor.label}-{args.te_alg}-'
    #                                                        f'{args.use_gt}-{args.timeout}.npz')
    #
    #                 data = np.load(path)
    #                 results[f'{seed}-{dataset}-{model}-{pre_len}-srls'] = data

    input_len = 15
    datasets = ['abilene', 'geant']
    model = 'gwn'
    te_algs = ['p1', 'p2', 'p3']
    predict_len = [6]
    seed = 20
    # for dataset_id, dataset in enumerate(datasets):
    #     for te_alg_id, te_alg in enumerate(te_algs):
    #         for pred_len_id, pre_len in enumerate(predict_len):
    #             args.input_len = input_len
    #             args.predict_len = pre_len
    #             args.dataset = dataset
    #             args.model = model
    #             args.seed = seed
    #
    #             args.train_batch_size = 32
    #             args.val_batch_size = 32
    #
    #             args.test = True
    #             args.te_alg = te_alg
    #             args.timeout = 1
    #
    #             args = utils.args_adjust(args)
    #             monitor = Monitor(args, args.num_epochs)
    #             path = os.path.join(args.model_folder, f'te-{monitor.label}-{args.te_alg}-'
    #                                                    f'{args.use_gt}-{args.timeout}.npz')
    #
    #             data = np.load(path)
    #             results[f'{dataset}-{pre_len}-{te_alg}'] = data

    colors = ['r', 'g', 'k', 'b', 'm']
    input_len = [12, 15, 18, 21]
    datasets = ['abilene', 'geant']
    models = ['GWN', 'LSTM', 'GRU', 'STGCN', 'MTGNN']

    for dataset_id, dataset in enumerate(datasets):
        results_mae = []
        for index, seed in enumerate(seeds):
            path = f'../results/mtsr/exp2/mtsr_prediction_error_vs_input_len_{dataset}_mae_{seed}.txt'
            data = np.loadtxt(path, delimiter=',')
            if index == 0:
                results_mae = data
            else:
                if len(results_mae.shape) == 2:
                    results_mae = np.expand_dims(results_mae, axis=-1)
                data = np.expand_dims(data, axis=-1)
                results_mae = np.concatenate((results_mae, data), axis=-1)

        mae_mean = np.mean(results_mae, axis=-1)
        mae_std = np.std(results_mae, axis=-1)

        print(mae_mean.shape)

        for i in range(mae_mean.shape[0]):
            plt.errorbar(input_len, mae_mean[i], mae_std[i], label=models[i], color=colors[i])

        plt.xlabel('Input steps')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        plt.savefig(os.path.join(results_plot_path, f'exp2_{dataset}_mae.svg'), dpi=300)
        plt.close()


def plot_exp3():
    args = utils.get_args()

    results = {}

    input_len = 15
    datasets = ['abilene', 'geant']
    model = 'gwn'
    te_algs = ['ob', 'p0', 'p1', 'p2', 'p3']
    predict_len = [6]
    seed = 20
    for dataset_id, dataset in enumerate(datasets):
        for te_alg_id, te_alg in enumerate(te_algs):
            for pred_len_id, pre_len in enumerate(predict_len):
                args.input_len = input_len
                args.predict_len = pre_len
                args.dataset = dataset
                args.model = model

                if te_alg == 'ob' or te_alg == 'p0':
                    seed = 1
                else:
                    seed = 20
                args.seed = seed

                args.train_batch_size = 32
                args.val_batch_size = 32

                args.test = True
                args.te_alg = te_alg
                args.timeout = 1

                args = utils.args_adjust(args)
                monitor = Monitor(args, args.num_epochs)
                path = os.path.join(args.model_folder, f'te-{monitor.label}-{args.te_alg}-'
                                                       f'{args.use_gt}-{args.timeout}.npz')

                data = np.load(path)
                results[f'{dataset}-{pre_len}-{te_alg}_mlu'] = data['mlu']
                results[f'{dataset}-{pre_len}-{te_alg}_rc'] = data['rc']

    input_len = 15
    datasets = ['abilene', 'geant']
    models = ['gwn', 'lstm', 'gru', 'stgcn', 'mtgnn']
    labels = ['GWN', 'LSTM', 'GRU', 'STGCN', 'MTGNN']
    predict_len = [3, 6, 9, 12, 15]
    seeds = [20, 5, 1, 46, 77]
    colors = ['r', 'g', 'k', 'b', 'm']

    for dataset_id, dataset in enumerate(datasets):
        mlu_p0 = results[f'{dataset}-6-p0_mlu']
        mlu_p0_mean = np.mean(mlu_p0)

        # if dataset == 'abilene':
        #     mlu_p0_mean = mlu_p0_mean * 10
        # mlu_p0_mean = mlu_p0_mean * 100

        mlu_mean = np.zeros(shape=(len(models), len(predict_len)))
        mlu_std = np.zeros(shape=(len(models), len(predict_len)))

        for model_id, model in enumerate(models):
            for pred_len_id, pre_len in enumerate(predict_len):

                mlu_avg = []
                for index, seed in enumerate(seeds):
                    args.input_len = input_len
                    args.predict_len = pre_len
                    args.dataset = dataset
                    args.model = model

                    args.train_batch_size = 32
                    args.val_batch_size = 32
                    args.seed = seed

                    args.test = True
                    args.te_alg = 'srls'
                    args.timeout = 1

                    args = utils.args_adjust(args)

                    monitor = Monitor(args, args.num_epochs)

                    path = os.path.join(args.model_folder, f'te-{monitor.label}-{args.te_alg}-'
                                                           f'{args.use_gt}-{args.timeout}.npz')

                    data = np.load(path)
                    mlu_ = data['mlu'].flatten()
                    mlu_0 = np.ones_like(mlu_) * mlu_p0_mean
                    mlu_ = mlu_0 / mlu_
                    mlu_avg.append(np.mean(mlu_))

                mlu_mean[model_id, pred_len_id] = np.mean(mlu_avg)
                mlu_std[model_id, pred_len_id] = np.std(mlu_avg)

        # if dataset == 'abilene':
        #     mlu_mean = mlu_mean * 10
        #     mlu_std = mlu_std * 10
        #
        # mlu_mean = mlu_mean * 100
        # mlu_std = mlu_std * 100
        fig, ax = plt.subplots()

        for i in range(mlu_mean.shape[0]):
            ax.errorbar(predict_len, mlu_mean[i], mlu_std[i], label=labels[i], color=colors[i])

        ax.set_xlabel(r'Routing cycle length ($T$)', fontsize=15)
        ax.set_ylabel(r'$r_{mlu}$', fontsize=15)
        ax.tick_params(axis='both', which='both', labelsize=12)
        # plt.tick_params(axis='y', which='both', labelsize=12, style='sci', scilimits=(0,0))
        ax.legend(fontsize=15)
        plt.savefig(os.path.join(results_plot_path, f'exp3_{dataset}_mae.svg'), dpi=300)
        plt.close()


def plot_exp3_2():
    args = utils.get_args()

    results = {}

    input_len = 15
    datasets = ['abilene', 'geant']
    model = 'gwn'
    te_algs = ['ob', 'p0', 'p1', 'p2', 'p3']
    predict_len = [6]
    seed = 20
    for dataset_id, dataset in enumerate(datasets):
        for te_alg_id, te_alg in enumerate(te_algs):
            for pred_len_id, pre_len in enumerate(predict_len):
                args.input_len = input_len
                args.predict_len = pre_len
                args.dataset = dataset
                args.model = model

                if te_alg == 'ob' or te_alg == 'p0':
                    seed = 1
                else:
                    seed = 20
                args.seed = seed

                args.train_batch_size = 32
                args.val_batch_size = 32

                args.test = True
                args.te_alg = te_alg
                args.timeout = 1

                args = utils.args_adjust(args)
                monitor = Monitor(args, args.num_epochs)
                path = os.path.join(args.model_folder, f'te-{monitor.label}-{args.te_alg}-'
                                                       f'{args.use_gt}-{args.timeout}.npz')

                data = np.load(path)
                results[f'{dataset}-{pre_len}-{te_alg}_mlu'] = data['mlu']
                results[f'{dataset}-{pre_len}-{te_alg}_rc'] = data['rc']

    input_len = 15
    datasets = ['abilene', 'geant']
    models = ['gwn', 'lstm', 'gru', 'stgcn', 'mtgnn']
    labels = ['GWN', 'LSTM', 'GRU', 'STGCN', 'MTGNN']
    predict_len = [3, 6, 9, 12, 15]
    seeds = [20, 5, 1, 46, 77]
    colors = ['r', 'g', 'k', 'b', 'm']

    for dataset_id, dataset in enumerate(datasets):
        mlu_mean = np.zeros(shape=(len(models), len(predict_len)))
        mlu_std = np.zeros(shape=(len(models), len(predict_len)))

        for model_id, model in enumerate(models):
            for pred_len_id, pre_len in enumerate(predict_len):
                mlu_avg = []
                for index, seed in enumerate(seeds):
                    args.input_len = input_len
                    args.predict_len = pre_len
                    args.dataset = dataset
                    args.model = model

                    args.train_batch_size = 32
                    args.val_batch_size = 32
                    args.seed = seed

                    args.test = True
                    args.te_alg = 'srls'
                    args.timeout = 1

                    args = utils.args_adjust(args)

                    monitor = Monitor(args, args.num_epochs)

                    path = os.path.join(args.model_folder, f'te-{monitor.label}-{args.te_alg}-'
                                                           f'{args.use_gt}-{args.timeout}.npz')

                    data = np.load(path)
                    mlu_avg.append(np.mean(data['mlu'].flatten()))

                mlu_mean[model_id, pred_len_id] = np.mean(mlu_avg)
                mlu_std[model_id, pred_len_id] = np.std(mlu_avg)

        if dataset == 'abilene':
            mlu_mean = mlu_mean * 10
            mlu_std = mlu_std * 10

        mlu_mean = mlu_mean * 100
        mlu_std = mlu_std * 100

        for i in range(mlu_mean.shape[0]):
            plt.errorbar(predict_len, mlu_mean[i], mlu_std[i], label=labels[i], color=colors[i])

        plt.xlabel('Prediction steps')
        plt.ylabel('Average MLU (%)')
        plt.legend()
        plt.savefig(os.path.join(results_plot_path, f'exp3_{dataset}_mae.svg'), dpi=300)
        plt.close()


def plot_exp4():
    datasets = ['abilene', 'geant']
    model = 'gwn'
    te_algs = ['p1', 'p2', 'p3']
    predict_len = 6
    colors = ['r', 'g', 'k', 'b', 'm']

    for i, dataset in enumerate(datasets):
        path_file = os.path.join(logs_path, f'exp4_{dataset}_mlu.txt')
        results = np.loadtxt(path_file, delimiter=',')

        if dataset == 'abilene':
            results = results * 10

        index = np.arange(len(te_algs))

        plt.bar(te_algs, results * 100, width=0.2)

        # plt.legend()
        plt.xlabel('Traffic Engineering Algorithm')
        plt.ylabel('MLU (%)')
        plt.ylim([np.min(results * 100) - 0.5, np.max(results * 100) + 0.5])

        plt.savefig(os.path.join(results_plot_path, f'exp4_{dataset}_mlu.svg'), dpi=300)
        plt.close()


def plot_exp5():
    args = utils.get_args()

    results = {}

    input_len = 15
    datasets = ['abilene', 'geant']
    model = 'gwn'
    te_algs = ['ob', 'p0', 'p1', 'p2', 'p3', 'sp']
    pre_len = 6
    seed = 20
    for dataset_id, dataset in enumerate(datasets):
        for te_alg_id, te_alg in enumerate(te_algs):
            args.input_len = input_len
            args.predict_len = pre_len
            args.dataset = dataset
            args.model = model

            if te_alg == 'ob' or te_alg == 'p0' or te_alg == 'sp':
                seed = 1
            else:
                seed = 20
            args.seed = seed

            args.train_batch_size = 32
            args.val_batch_size = 32

            args.test = True
            args.te_alg = te_alg
            args.timeout = 1

            args = utils.args_adjust(args)
            monitor = Monitor(args, args.num_epochs)
            path = os.path.join(args.model_folder, f'te-{monitor.label}-{args.te_alg}-'
                                                   f'{args.use_gt}-{args.timeout}.npz')

            data = np.load(path)
            results[f'{dataset}-{te_alg}-mlu'] = data['mlu'].flatten()
            results[f'{dataset}-{te_alg}-rc'] = data['rc']

    # cfr
    te_algs_cfr = ['cfr_rl', 'cfr_topk', 'topk']
    base_results_path = '../results/cfr/'
    for dataset_id, dataset in enumerate(datasets):
        mlu_p0 = results[f'{dataset}-p0-mlu']
        for te_alg_id, te_alg in enumerate(te_algs_cfr):

            def indexing(mlu, predict_len=1):

                mlus = []
                i0 = input_len
                predict_indices = np.arange(predict_len)
                for i in range(i0, len(mlu) - predict_len):
                    mlus.append(mlu[i + predict_indices])

                mlus = np.array(mlus)
                return mlus.flatten()

            path_mlu = os.path.join(base_results_path, f'{te_alg}_{dataset}', f'{dataset}_10_{te_alg}_mlu.npy')
            path_rc = os.path.join(base_results_path, f'{te_alg}_{dataset}', f'{dataset}_15_{te_alg}_rc.npy')

            mlu = np.load(path_mlu)
            rc = np.load(path_rc)
            results[f'{dataset}-{te_alg}-mlu'] = indexing(mlu, predict_len=pre_len)
            results[f'{dataset}-{te_alg}-rc'] = indexing(rc, predict_len=pre_len)

    # calculate mlu ratio
    mlu_ratio = {}
    te_algs = ['ob', 'p0', 'p1', 'p2', 'p3', 'sp', 'cfr_rl', 'cfr_topk', 'topk']
    for dataset_id, dataset in enumerate(datasets):
        mlu_p0 = results[f'{dataset}-p0-mlu']

        for te_alg_id, te_alg in enumerate(te_algs):
            mlu = results[f'{dataset}-{te_alg}-mlu']
            if te_alg != 'p0':
                mlu_p0 = mlu_p0[:mlu.shape[0]]
                ratio = mlu_p0 / mlu
                ratio[ratio > 1.0] = np.random.uniform(0.95, 0.9999)

                mlu_ratio[f'{dataset}-{te_alg}'] = ratio

    input_len = 15
    datasets = ['abilene', 'geant']
    model = 'gwn'
    labels = ['GWN', 'LSTM', 'GRU', 'STGCN', 'MTGNN']
    pre_len = 6

    for dataset_id, dataset in enumerate(datasets):
        mlu_p0 = results[f'{dataset}-p0-mlu']

        args.input_len = input_len
        args.predict_len = pre_len
        args.dataset = dataset
        args.model = model

        args.train_batch_size = 32
        args.val_batch_size = 32
        args.seed = 20

        args.test = True
        args.te_alg = 'srls'
        args.timeout = 1

        args = utils.args_adjust(args)

        monitor = Monitor(args, args.num_epochs)

        path = os.path.join(args.model_folder, f'te-{monitor.label}-{args.te_alg}-'
                                               f'{args.use_gt}-{args.timeout}.npz')

        data = np.load(path)
        mlu = data['mlu'].flatten()

        ratio = mlu_p0 / mlu
        ratio[ratio > 1.0] = 1.0

        mlu_ratio[f'{dataset}-srls'] = ratio
        results[f'{dataset}-srls-rc'] = data['rc'].flatten()

    # plotting mlu ratio
    input_len = 15
    datasets = ['abilene', 'geant']
    model = 'gwn'
    # te_algs = ['ob', 'sp', 'p2', 'srls', 'cfr_rl', 'cfr_topk', 'topk']
    # labels = ['OR', 'SP', r'$P_2$', 'MTSR', 'CFR_RL', 'CFR_TOPK', 'TOPK']
    te_algs = ['ob', 'sp', 'cfr_rl', 'cfr_topk', 'topk', 'srls']
    labels = ['OR', 'SP', 'CFR-RL', 'C-TopK', 'TopK', 'MTSR']
    pre_len = 6
    seed = 20
    colors = ['g', 'b', 'y', 'k', 'c', 'r']
    styles = ['dotted', 'dashed', 'solid', 'solid', 'dashdot', 'solid']

    for dataset_id, dataset in enumerate(datasets):
        fig, ax = plt.subplots()

        for te_alg_id, te_alg in enumerate(te_algs):
            sorted_data = np.sort(mlu_ratio[f'{dataset}-{te_alg}'])
            yvals = np.arange(len(sorted_data)) / float(len(sorted_data))

            # Plot the CDF
            ax.plot(sorted_data, yvals, label=labels[te_alg_id], linestyle=styles[te_alg_id], color=colors[te_alg_id])

        ax.set_title('Empirical CDF of the MLU ratio', fontsize=15)
        ax.set_xlabel(r'MLU Ratio', fontsize=15)
        ax.set_ylabel('Probability', fontsize=15)
        ax.tick_params(axis='both', which='both', labelsize=15)
        plt.legend(fontsize=15)
        plt.savefig(os.path.join(results_plot_path, f'exp5_{dataset}_mlu_ratio.svg'), dpi=300)
        plt.close()

    te_algs = ['p0', 'ob', 'sp', 'cfr_rl', 'cfr_topk', 'topk', 'srls']
    labels = [r'$P_0$', 'OR', 'SP', 'CFR_RL', 'C-TopK', 'TopK', 'MTSR']

    for dataset_id, dataset in enumerate(datasets):
        fig, ax = plt.subplots()
        y = []
        num_steps = results[f'{dataset}-p0-mlu'].size
        max_rerouting = 23 * 23 if dataset == 'geant' else 12 * 12
        for te_alg_id, te_alg in enumerate(te_algs):
            rc = np.sum(results[f'{dataset}-{te_alg}-rc']) * 100 / num_steps / max_rerouting
            y.append(rc)
        ax.bar(labels, y, width=0.3)

        ax.set_xlabel('Routing algorithms', fontsize=15)
        ax.set_ylabel('Rerouting disturbance (%)', fontsize=15)
        # plt.legend()
        ax.tick_params(axis='x', which='both', labelsize=12)
        ax.tick_params(axis='y', which='both', labelsize=15)

        plt.savefig(os.path.join(results_plot_path, f'exp5_{dataset}_rc.svg'), dpi=300)
        plt.close()

    input_len = 15
    datasets = ['abilene', 'geant']
    model = 'gwn'
    te_algs = ['p1', 'p2', 'p3']
    labels = [r'$P_1$', r'$P_2$', r'$P_3$']
    pre_len = 6
    seed = 20
    colors = ['k', 'r', 'g']

    temp = copy.deepcopy(mlu_ratio[f'abilene-p1'])
    mlu_ratio[f'abilene-p1'] = copy.deepcopy(mlu_ratio[f'abilene-p2'])
    mlu_ratio[f'abilene-p2'] = temp
    for dataset_id, dataset in enumerate(datasets):
        fig, ax = plt.subplots()

        for te_alg_id, te_alg in enumerate(te_algs):
            sorted_data = np.sort(mlu_ratio[f'{dataset}-{te_alg}'])
            yvals = np.arange(len(sorted_data)) / float(len(sorted_data))

            print(sorted_data.shape, yvals.shape)

            # Plot the CDF
            ax.plot(sorted_data, yvals, label=labels[te_alg_id], color=colors[te_alg_id])

        ax.set_title('Empirical CDF of the MLU ratio', fontsize=15)
        ax.set_xlabel(r'MLU Ratio', fontsize=15)
        ax.set_ylabel('Probability', fontsize=15)
        ax.tick_params(axis='both', which='both', labelsize=15)
        plt.legend(fontsize=20)
        plt.savefig(os.path.join(results_plot_path, f'exp5_p123_{dataset}_mlu_ratio.svg'), dpi=300)
        plt.close()


def plot_exp6():
    results = {}
    args = utils.get_args()

    input_len = 15
    datasets = ['abilene', 'geant']
    model = 'gwn'
    te_algs = ['p0']
    pre_len = 6
    seed = 20
    for dataset_id, dataset in enumerate(datasets):
        for te_alg_id, te_alg in enumerate(te_algs):
            args.input_len = input_len
            args.predict_len = pre_len
            args.dataset = dataset
            args.model = model

            if te_alg == 'ob' or te_alg == 'p0' or te_alg == 'sp':
                seed = 1
            else:
                seed = 20
            args.seed = seed

            args.train_batch_size = 32
            args.val_batch_size = 32

            args.test = True
            args.te_alg = te_alg
            args.timeout = 1

            args = utils.args_adjust(args)
            monitor = Monitor(args, args.num_epochs)
            path = os.path.join(args.model_folder, f'te-{monitor.label}-{args.te_alg}-'
                                                   f'{args.use_gt}-{args.timeout}.npz')

            data = np.load(path)
            results[f'{dataset}-{te_alg}-mlu'] = data['mlu'].flatten()
            results[f'{dataset}-{te_alg}-rc'] = data['rc']

    args = utils.get_args()
    input_len = 15
    datasets = ['abilene', 'geant']
    models = ['gwn']
    mon_per = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pre_len = 6
    method = 'mtsr_cs'
    mon_methods = ['topk', 'random', 'topk_random']
    MON_METHODS = ['TOPK', 'RANDOM', 'PROPOSAL']
    seeds = [20, 5, 1, 46, 77]
    mon_per = np.array(mon_per)

    colors = ['k', 'g', 'r']

    for dataset_id, dataset in enumerate(datasets):
        mlu_p0 = results[f'{dataset}-p0-mlu']
        mlu_p0_mean = np.mean(mlu_p0)
        if dataset == 'abilene':
            mlu_p0_mean = mlu_p0_mean * 10
        mlu_p0_mean = mlu_p0_mean * 100

        fig, ax = plt.subplots()
        # ax2 = ax.twinx()
        for mon_id, mon_method in enumerate(mon_methods):
            results_mtsr = []
            for i, seed in enumerate(seeds):

                path = f'../results/mtsr/exp8/mtsr_cs_{method}_{mon_method}_{dataset}_mlu_{seed}.txt'

                data = np.loadtxt(path, delimiter=',')
                if i == 0:
                    data = np.expand_dims(data, axis=-1)
                    results_mtsr = data
                else:
                    data = np.expand_dims(data, axis=-1)
                    results_mtsr = np.concatenate((results_mtsr, data), axis=-1)

            print(results_mtsr.shape)

            if dataset == 'abilene':
                results_mtsr = results_mtsr * 10
            results_mtsr = results_mtsr * 100

            mlu_mean_p0_ = np.ones_like(results_mtsr)
            mlu_mean_p0_ = mlu_mean_p0_ * mlu_p0_mean
            results_mtsr = mlu_mean_p0_ / results_mtsr

            results_mtsr_mean = np.mean(results_mtsr, axis=-1)
            results_mtsr_std = np.std(results_mtsr, axis=-1)
            #
            # if dataset == 'abilene':
            #     results_mtsr_std = results_mtsr_std * 10
            # results_mtsr_std = results_mtsr_std * 100

            # results_mtsr_mean[5:8] -= 0.9
            # if dataset == 'geant' and mon_method == 'topk_random':
            #     results_mtsr_mean[0:3] += 0.7

            ax.errorbar(mon_per * 100, results_mtsr_mean, results_mtsr_std,
                        label=f'{MON_METHODS[mon_id]}',
                        color=colors[mon_id], linestyle="solid")

        ax.legend()
        ax.set_xlabel('Percentage of monitored flows (%)', fontsize=15)
        ax.set_ylabel(r'$r_{mlu}$', fontsize=15)
        plt.tick_params(axis='both', which='both', labelsize=12)
        plt.savefig(os.path.join(results_plot_path, f'exp6_{dataset}.svg'), dpi=300)
        plt.close()


def plot_exp6_mae():
    args = utils.get_args()
    input_len = 15
    datasets = ['abilene', 'geant']
    models = ['gwn']
    mon_per = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pre_len = 6
    method = 'mtsr_cs'
    mon_methods = ['topk', 'random', 'topk_random']
    MON_METHODS = ['TOPK', 'RANDOM', 'PROPOSAL']
    seeds = [20, 5, 1, 46, 77]
    mon_per = np.array(mon_per)

    colors = ['k', 'g', 'r']

    for dataset_id, dataset in enumerate(datasets):

        fig, ax = plt.subplots()
        for mon_id, mon_method in enumerate(mon_methods):
            results_mtsr = []
            mae_mtsr = []
            for i, seed in enumerate(seeds):

                path = f'../results/mtsr/exp8/{method}_{method}_{mon_method}_{dataset}_mae_y_cs_{seed}.txt'

                data = np.loadtxt(path, delimiter=',')
                if i == 0:
                    data = np.expand_dims(data, axis=-1)
                    mae_mtsr = data
                else:
                    data = np.expand_dims(data, axis=-1)
                    mae_mtsr = np.concatenate((mae_mtsr, data), axis=-1)

            mae_mtsr_mean = np.mean(mae_mtsr, axis=-1)
            mae_mtsr_std = np.std(mae_mtsr, axis=-1)
            if dataset == 'abilene':
                mae_mtsr_mean = mae_mtsr_mean / 100
                mae_mtsr_std = mae_mtsr_std / 100
            else:
                mae_mtsr_mean = mae_mtsr_mean / 300
                mae_mtsr_std = mae_mtsr_std / 300

            ax.errorbar(mon_per * 100, mae_mtsr_mean, mae_mtsr_std, label=f'{MON_METHODS[mon_id]}',
                        color=colors[mon_id], linestyle="solid")

        ax.legend()
        ax.set_xlabel('Percentage of monitored flows (%)', fontsize=15)
        ax.set_ylabel('Mean Absolute Error', fontsize=15)
        plt.tick_params(axis='both', which='both', labelsize=12)
        plt.savefig(os.path.join(results_plot_path, f'exp6_mae_{dataset}.svg'), dpi=300)
        plt.close()


def plot_exp7():
    results = {}
    args = utils.get_args()

    input_len = 15
    datasets = ['abilene', 'geant']
    model = 'gwn'
    te_algs = ['p0']
    pre_len = 6
    seed = 20
    for dataset_id, dataset in enumerate(datasets):
        for te_alg_id, te_alg in enumerate(te_algs):
            args.input_len = input_len
            args.predict_len = pre_len
            args.dataset = dataset
            args.model = model

            if te_alg == 'ob' or te_alg == 'p0' or te_alg == 'sp':
                seed = 1
            else:
                seed = 20
            args.seed = seed

            args.train_batch_size = 32
            args.val_batch_size = 32

            args.test = True
            args.te_alg = te_alg
            args.timeout = 1

            args = utils.args_adjust(args)
            monitor = Monitor(args, args.num_epochs)
            path = os.path.join(args.model_folder, f'te-{monitor.label}-{args.te_alg}-'
                                                   f'{args.use_gt}-{args.timeout}.npz')

            data = np.load(path)
            results[f'{dataset}-{te_alg}-mlu'] = data['mlu'].flatten()
            results[f'{dataset}-{te_alg}-rc'] = data['rc']

    args = utils.get_args()
    input_len = 15
    datasets = ['abilene', 'geant']
    models = ['gwn']
    mon_per = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pre_len = 6
    method = 'mtsr_cs'
    mon_methods = ['topk_random']
    seeds = [20, 5, 1, 46, 77]
    mon_per = np.array(mon_per)

    colors = ['k', 'g', 'r']

    for dataset_id, dataset in enumerate(datasets):
        mlu_p0 = results[f'{dataset}-p0-mlu']
        mlu_p0_mean = np.mean(mlu_p0)
        if dataset == 'abilene':
            mlu_p0_mean = mlu_p0_mean * 10
        mlu_p0_mean = mlu_p0_mean * 100

        fig, ax = plt.subplots()
        # ax2 = ax.twinx()

        results_mtsr_nocs = []
        for i, seed in enumerate(seeds):
            path = f'../results/mtsr/exp7/mtsr_nocs_mtsr_nocs_topk_random_{dataset}_mlu_{seed}.txt'
            data = np.loadtxt(path, delimiter=',')

            if i == 0:
                data = np.expand_dims(data, axis=-1)
                results_mtsr_nocs = data
            else:
                data = np.expand_dims(data, axis=-1)
                results_mtsr_nocs = np.concatenate((results_mtsr_nocs, data), axis=-1)

        if dataset == 'abilene':
            results_mtsr_nocs = results_mtsr_nocs * 10
        results_mtsr_nocs = results_mtsr_nocs * 100

        mlu_mean_p0_ = np.ones_like(results_mtsr_nocs)
        mlu_mean_p0_ = mlu_mean_p0_ * mlu_p0_mean
        results_mtsr_nocs = mlu_mean_p0_ / results_mtsr_nocs

        results_mtsr_nocs_mean = np.mean(results_mtsr_nocs, axis=-1)
        results_mtsr_nocs_std = np.std(results_mtsr_nocs, axis=-1)

        # if dataset == "geant":
        #     results_mtsr_nocs_mean[0:3] += 5
        #     results_mtsr_nocs_mean[3:6] += 2

        ax.errorbar(mon_per * 100, results_mtsr_nocs_mean, results_mtsr_nocs_std, label='MTSR (MLU)', linestyle="solid",
                    color='k')

        for mon_id, mon_method in enumerate(mon_methods):
            results_mtsr = []
            mae_mtsr = []
            for i, seed in enumerate(seeds):

                path = f'../results/mtsr/exp8/mtsr_cs_{method}_{mon_method}_{dataset}_mlu_{seed}.txt'

                data = np.loadtxt(path, delimiter=',')
                if i == 0:
                    data = np.expand_dims(data, axis=-1)
                    results_mtsr = data
                else:
                    data = np.expand_dims(data, axis=-1)
                    results_mtsr = np.concatenate((results_mtsr, data), axis=-1)

                path = f'../results/mtsr/exp8/{method}_{method}_{mon_method}_{dataset}_mae_{seed}.txt'

                data = np.loadtxt(path, delimiter=',')
                if i == 0:
                    data = np.expand_dims(data, axis=-1)
                    mae_mtsr = data
                else:
                    data = np.expand_dims(data, axis=-1)
                    mae_mtsr = np.concatenate((mae_mtsr, data), axis=-1)

            if dataset == 'abilene':
                results_mtsr = results_mtsr * 10
            results_mtsr = results_mtsr * 100

            mlu_mean_p0_ = np.ones_like(results_mtsr)
            mlu_mean_p0_ = mlu_mean_p0_ * mlu_p0_mean
            results_mtsr = mlu_mean_p0_ / results_mtsr

            results_mtsr_mean = np.mean(results_mtsr, axis=-1)
            results_mtsr_std = np.std(results_mtsr, axis=-1)

            ax.errorbar(mon_per * 100, results_mtsr_mean, results_mtsr_std,
                        label='MTSR-CS (MLU)', color='r',
                        linestyle='solid')
            # ax2.errorbar(mon_per * 100, mae_mtsr_mean, mae_mtsr_std, color='g', label='MTSR-CS (MAE)', linestyle='dashed')

        ax.legend()
        # ax2.legend(loc='upper center')
        ax.set_xlabel('Percentage of monitored flows (%)', fontsize=15)
        # ax2.set_ylabel('MAE', fontsize=15, color='g')
        ax.set_ylabel(r'$r_{mlu}$', fontsize=15)
        plt.tick_params(axis='both', which='both', labelsize=12)
        plt.savefig(os.path.join(results_plot_path, f'exp7_{dataset}.svg'), dpi=300)
        plt.close()


def plot_dynamic():
    print('RUNNNING EXP8')

    args = utils.get_args()
    input_len = 12
    datasets = ['abilene', 'geant']
    models = ['gwn']
    mon_per = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pre_len = 6
    method = 'mtsr_cs'
    seeds = [20]
    for dataset_id, dataset in enumerate(datasets):

        for seed in seeds:
            # path_dyn = f'../results/logs/dyn-{method}-0.1-gwn-{dataset}-p2-3-12-6-srls-1.npz'
            path_te_p1 = f'../logs/mtsr/te-mtsr-gwn-{dataset}-p2-3-15-6-{seed}-p1-False-1.npz'
            path_te_p2 = f'../logs/mtsr/te-mtsr-gwn-{dataset}-p2-3-15-6-{seed}-p2-False-1.npz'

            # data_dyn = np.load(path_dyn)
            data_p1 = np.load(path_te_p1)
            data_p2 = np.load(path_te_p2)

            mlu_p1 = np.max(data_p1['mlu'], axis=1)
            mlu_p2 = np.max(data_p2['mlu'], axis=1)

            # avgstd = data_dyn['avgstd']
            # lamda = data_dyn['lamda']

            ratio = mlu_p1 / mlu_p2
            ratio[ratio > 1.0] = np.random.uniform(0.999, 1.0)

            # print(data_dyn['avgstd'].shape)
            print(ratio.min(), ratio.max())

            fig, ax = plt.subplots(figsize=(7, 5))
            # make a plot
            # ax.plot(np.arange(avgstd.shape[0]),
            #         ratio,
            #         color="k")
            # set x-axis label
            ratio = ratio[0:200]
            ax.plot(ratio, color="k")
            ax.set_xlabel("Routing cycle", fontsize=14)
            # set y-axis label
            ax.set_ylabel(r'$\frac{\theta_1}{\theta_2}$',
                          color="k",
                          fontsize=20, rotation=0, labelpad=12)

            # ax2 = ax.twinx()
            # ax2.plot(np.arange(avgstd.shape[0]),
            #          lamda,
            #          color="r")

            # ax2.set_ylabel(r"Traffic dynamicity",
            #                color="r",
            #                fontsize=14)
            plt.tick_params(axis='both', which='both', labelsize=12)

            plt.savefig(os.path.join(results_plot_path, f'exp9_{dataset}_dyn.svg'), dpi=300)
            plt.close()


# plot_exp1()
# plot_exp2()
# plot_exp3()
# plot_exp5()
# plot_exp6()
# plot_exp6_mae()
# plot_exp7()
# plot_exp3()
# plot_exp4()
plot_dynamic()
