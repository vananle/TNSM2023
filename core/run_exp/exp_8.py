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

def exp_8(datasets, models, mon_per, input_len, predict_len, colors, label_models, seeds):
    print('RUNNNING EXP8')

    args = utils.get_args()
    args.data_folder = '../../data'
    args.tensorboard_folder = '../../logs/core/'
    args.csv_folder = '../../data/csv/'
    args.model_folder = '../../logs/core/'

    t1 = time.time()
    input_len = 15
    models = ['gwn']
    mon_per = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pre_len = 6
    method = 'mtsr_cs'
    mon_method = 'topk_random'  # random, topk, topk_random, topk_per_node

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

            os.makedirs('../../results/core/exp8', exist_ok=True)
            for k, v in results.items():
                np.savetxt(f'../../results/core/exp8/mtsr_cs_{method}_{mon_method}_{dataset}_{k}_{seed}.txt', results[k],
                           delimiter=',')

    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())



def plot_exp8(datasets, models, input_len, predict_len, colors, label_models, seeds):
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

                path = f'../results/core/exp8/{method}_{method}_{mon_method}_{dataset}_mae_y_cs_{seed}.txt'

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



if __name__ == "__main__":

    input_len = 15
    models = ['gwn']
    mon_per = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    predict_len = 6
    method = 'mtsr_cs'
    mon_method = 'topk_random'  # random, topk, topk_random, topk_per_node

    datasets = ['germany', 'gnnet-40']
    seeds = [20, 5, 1, 46, 77]
    colors = ['r', 'g', 'k', 'b', 'm']
    label_models = ['GWN', 'LSTM', 'GRU', 'STGCN', 'MTGNN']

    # exp_1(datasets, models, input_len, predict_len, seeds)
    plot_exp8(datasets, models, input_len, predict_len, colors, label_models, seeds)
