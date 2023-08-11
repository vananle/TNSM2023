import sys

import matplotlib.pyplot as plt

sys.path.append('../../')

from core.run_exp.base_runner import *

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

logs_path = '../../results/core/'
results_plot_path = '../../results_plot/core'
if not os.path.exists(results_plot_path):
    os.makedirs(results_plot_path)


def exp_7():
    print('RUNNNING EXP 7')

    args = utils.get_args()
    args.data_folder = '../../data'
    args.tensorboard_folder = '../../logs/core/'
    args.csv_folder = '../../data/csv/'
    args.model_folder = '../../logs/core/'

    t1 = time.time()
    input_len = 15
    # datasets = ['abilene', 'geant']
    datasets = ['germany', 'gnnet-40']
    models = ['gwn']
    mon_per = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pre_len = 6
    method = 'mtsr_nocs'
    mon_method = 'random'  # random, topk, topk_random, topk_per_node

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

            os.makedirs(name=os.path.join(logs_path, 'exp7'), exist_ok=True)
            for k, v in results.items():
                np.savetxt(f'../../results/core/exp7/mtsr_nocs_{method}_{mon_method}_{dataset}_{k}_{seed}.txt',
                           results[k],
                           delimiter=',')

    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())


def plot_exp7():
    results = {}
    args = utils.get_args()
    args.data_folder = '../../data'
    args.tensorboard_folder = '../../logs/core/'
    args.csv_folder = '../../data/csv/'
    args.model_folder = '../../logs/core/'

    input_len = 15
    # datasets = ['abilene', 'geant']
    datasets = ['germany', 'gnnet-40']
    model = 'gwn'
    te_algs = ['p2']
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
    datasets = ['germany', 'gnnet-40']
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
            path = os.path.join(logs_path, f'exp7/mtsr_nocs_mtsr_nocs_topk_random_{dataset}_mlu_{seed}.txt')
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

        ax.errorbar(mon_per * 100, results_mtsr_nocs_mean, results_mtsr_nocs_std, label='MTSR (MLU)', linestyle="solid",
                    color='k')

        for mon_id, mon_method in enumerate(mon_methods):
            results_mtsr = []
            mae_mtsr = []
            for i, seed in enumerate(seeds):

                path = os.path.join(logs_path, f'exp8/mtsr_cs_{method}_{mon_method}_{dataset}_mlu_{seed}.txt')

                data = np.loadtxt(path, delimiter=',')
                if i == 0:
                    data = np.expand_dims(data, axis=-1)
                    results_mtsr = data
                else:
                    data = np.expand_dims(data, axis=-1)
                    results_mtsr = np.concatenate((results_mtsr, data), axis=-1)

                path = os.path.join(logs_path, f'exp8/{method}_{method}_{mon_method}_{dataset}_mae_{seed}.txt')

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

        ax.legend()
        # ax2.legend(loc='upper center')
        ax.set_xlabel('Percentage of monitored flows (%)', fontsize=15)
        # ax2.set_ylabel('MAE', fontsize=15, color='g')
        ax.set_ylabel(r'$r_{mlu}$', fontsize=15)
        plt.tick_params(axis='both', which='both', labelsize=12)
        plt.savefig(os.path.join(results_plot_path, f'exp7_{dataset}.svg'), dpi=300)
        plt.close()


if __name__ == "__main__":
    input_len = 15
    models = ['gwn']
    mon_per = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    predict_len = 6
    method = 'mtsr_cs'
    mon_method = 'random'  # random, topk, topk_random, topk_per_node

    datasets = ['gnnet-40']
    seeds = [20, 5, 1, 46, 77]
    colors = ['k', 'g', 'r', 'b', 'm']
    label_models = ['TOPK', 'RANDOM', 'PROPOSAL']

    # plot_exp8(datasets, mon_per, colors, label_models, seeds)
    plot_exp7()

    # exp_7()
