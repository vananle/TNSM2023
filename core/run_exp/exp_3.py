import itertools
import sys

from joblib import Parallel, delayed

sys.path.append('../../')

from core.run_exp.base_runner import *

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


def exp_3():
    print('RUN EXP 3')
    args = utils.get_args()
    args.data_folder = '../../data'
    args.tensorboard_folder = '../../logs/core/'
    args.csv_folder = '../../data/csv/'
    args.model_folder = '../../logs/core/'

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

            os.makedirs('../../results/core/exp3/', exist_ok=True)
            for k, v in results.items():
                np.savetxt(f'../../results/core/exp3/exp3_{dataset}_{k}_{seed}.txt', results[k], delimiter=',')

    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())


exp_3()
