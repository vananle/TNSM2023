import sys

sys.path.append('../../')

from core.run_scripts.base_runner import *

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

    exp_1(datasets, models, input_len, predict_len, seeds)
