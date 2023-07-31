import sys

sys.path.append('../../../')

from core.run_exp.run_exp import *

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


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


exp_2()
