import sys

sys.path.append('../../../')

from .run_exp import *

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


def exp_10():
    print('RUNNNING EXP10 VAE')

    args = utils.get_args()
    t1 = time.time()
    input_len = 15
    datasets = ['abilene', 'geant']
    models = ['vae']
    pre_len = 6
    method = 'mtsr_cs'
    mon_method = 'random'

    seeds = [20, 5, 1, 46, 77]
    # seeds = [77]

    for seed in seeds:

        for dataset_id, dataset in enumerate(datasets):
            results = {'mlu': [],
                       'rc': []}

            for model_id, model in enumerate(models):
                args.input_len = input_len
                args.predict_len = pre_len
                args.dataset = dataset
                args.model = model
                args.mon_method = mon_method
                args.method = method
                args.seed = seed

                args.train_batch_size = 32
                args.val_batch_size = 32

                args = utils.args_adjust(args)
                args.test = True
                args.te_alg = 'srls'
                args.timeout = 1

                _, mlu, rc = run_exp(args)
                results['mlu'].append(np.mean(mlu))
                results['rc'].append(np.mean(mlu))

            os.makedirs('../results/core/exp10/', exist_ok=True)
            for k, v in results.items():
                np.savetxt(f'../results/core/exp10/mtsr_cs_vae_{dataset}_{k}_{seed}.txt', results[k], delimiter=',')

    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())


exp_10()
