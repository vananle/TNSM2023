import sys

sys.path.append('../../')

from core.run_exp.base_runner import *

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


def exp_11(datasets, models, input_len):
    args = utils.get_args()

    args.data_folder = '../../data'
    args.tensorboard_folder = '../../logs/core/'
    args.csv_folder = '../../data/csv/'
    args.model_folder = '../../logs/core/'


    t1 = time.time()
    te_algs = ['ob']
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
                args.model = models
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

        os.makedirs('../../results/core/exp11/', exist_ok=True)
        for k, v in results.items():
            np.savetxt(f'../../results/core/exp11/exp11_{dataset}_{k}.txt', results[k], delimiter=',')

    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())


if __name__ == "__main__":

    input_len = 15
    # datasets = ['abilene', 'geant']
    datasets = ['germany', 'gnnet-40']
    models = 'gwn'
    predict_len = [3, 6, 9, 12]
    seeds = [20, 5, 1, 46, 77]

    exp_11(datasets, models, input_len)
