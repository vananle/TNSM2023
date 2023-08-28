import sys

sys.path.append('../../')

from core.run_scripts.base_runner import *

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

logs_path = '../../results/core/'
results_plot_path = '../../results_plot/core'
if not os.path.exists(results_plot_path):
    os.makedirs(results_plot_path)


def exp_8(datasets, mon_method, models, mon_per, input_len, predict_len, seeds):
    print('RUNNNING EXP8 mtsr_cs')

    args = utils.get_args()
    args.data_folder = '../../data'
    args.tensorboard_folder = '../../logs/core/'
    args.csv_folder = '../../data/csv/'
    args.model_folder = '../../logs/core/'

    t1 = time.time()
    method = 'mtsr_cs'

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
                    args.predict_len = predict_len
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
            #         for k, v in metrics.items():
            #             results[k][model_id, mon_p_id] = v
            #         results['mlu'][model_id, mon_p_id] = np.mean(mlu)
            #         results['rc'][model_id, mon_p_id] = np.mean(rc)
            #         print(dataset, model, predict_len, metrics)
            #
            # os.makedirs(os.path.join(logs_path, 'exp8'), exist_ok=True)
            # for k, v in results.items():
            #     np.savetxt(os.path.join(logs_path, f'exp8/mtsr_cs_{method}_{mon_method}_{dataset}_{k}_{seed}.txt'),
            #                results[k],
            #                delimiter=',')

    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())


if __name__ == "__main__":
    input_len = 15
    models = ['gwn']
    mon_per = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    predict_len = 6
    method = 'mtsr_cs'
    mon_method = 'topk_random'  # random, topk, topk_random, topk_per_node

    datasets = ['abilene']
    # seeds = [20, 5, 1, 46, 77]
    seeds = [20]
    colors = ['k', 'g', 'r', 'b', 'm']
    label_models = ['TOPK', 'RANDOM', 'PROPOSAL']

    exp_8(datasets, mon_method, models, mon_per, input_len, predict_len, seeds)
