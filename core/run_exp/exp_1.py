import sys

sys.path.append('../../')
from matplotlib import pyplot as plt

from core.run_exp.base_runner import *

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

logs_path = '../../results/core/'
results_plot_path = '../../results_plot/core'
if not os.path.exists(results_plot_path):
    os.makedirs(results_plot_path)


def exp_1(datasets, models, input_len, predict_len, seeds):
    print('|----- RUN EXP 1: TRAFFIC PREDICTION WITH DIFFERENT PRED_LEN ----')
    args = utils.get_args()

    args.data_folder = '../../data'
    args.tensorboard_folder = '../../logs/core/'
    args.csv_folder = '../../data/csv/'
    args.model_folder = '../../logs/core/'

    t1 = time.time()
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
                    args.timeout = 60
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

def plot_runtime():

    values = [0.050559921698136764, 0.13788055289875378, 0.26901960372924805, 0.49418334166208905]
    fig, ax = plt.subplots(figsize=(9, 5))
    datasets = ['144 (Abilene)', '484 flows (Geant)', '1600 (Gnnet-40)', '2500 (Germany)']

    plt.bar(datasets, values)

    ax.set_xlabel(r'Number of flows', fontsize=15)
    ax.set_ylabel('Prediction time (second)', fontsize=15)
    ax.tick_params(axis='both', which='both', labelsize=12)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # plt.legend(fontsize=15)
    plt.savefig(os.path.join(results_plot_path, f'run_time.svg'), dpi=300)
    plt.close()


def plot_exp1(datasets, models, input_len, predict_len, colors, label_models, seeds):
    args = utils.get_args()
    results = {}
    args.data_folder = '../../data'
    args.tensorboard_folder = '../../logs/core/'
    args.csv_folder = '../../data/csv/'
    args.model_folder = '../../logs/core/'

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

    for dataset_id, dataset in enumerate(datasets):
        results_mae = []
        for index, seed in enumerate(seeds):
            path = f'../../results/core/exp1/mtsr_prediction_error_vs_prediction_len_{dataset}_mae_{seed}.txt'
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
            ax.errorbar(predict_len, mae_mean[i], mae_std[i], label=label_models[i], color=colors[i])

        ax.set_xlabel(r'Routing cycle length ($T$)', fontsize=15)
        ax.set_ylabel('Mean Absolute Error', fontsize=15)
        ax.tick_params(axis='both', which='both', labelsize=12)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.legend(fontsize=15)
        plt.savefig(os.path.join(results_plot_path, f'exp1_{dataset}_mae.svg'), dpi=300)
        plt.close()


if __name__ == "__main__":

    input_len = 15
    # datasets = ['abilene', 'geant', 'germany', 'gnnet-40']
    datasets = ['gnnet-40']
    # models = ['gwn', 'lstm', 'gru', 'stgcn', 'mtgnn']
    models = ['gwn']
    # predict_len = [3, 6, 9, 12]
    predict_len = [6]
    # seeds = [20, 5, 1, 46, 77]
    seeds = [20]
    colors = ['r', 'g', 'k', 'b', 'm']
    label_models = ['GWN', 'LSTM', 'GRU', 'STGCN', 'MTGNN']

    # exp_1(datasets, models, input_len, predict_len, seeds)
    # plot_exp1(datasets, models, input_len, predict_len, colors, label_models, seeds)
    plot_runtime()
