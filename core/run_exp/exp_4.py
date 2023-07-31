import sys

sys.path.append('../../../')

from .run_exp import *

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


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


exp_4()
