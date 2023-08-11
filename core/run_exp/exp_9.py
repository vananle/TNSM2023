import sys

sys.path.append('../../')

from core.run_exp.base_runner import *

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

logs_path = '../../results/core/'
results_plot_path = '../../results_plot/core'
if not os.path.exists(results_plot_path):
    os.makedirs(results_plot_path)


def exp_9(datasets, models, input_len, predict_len, seeds):
    print('|----- RUN EXP 1: TRAFFIC PREDICTION WITH DIFFERENT PRED_LEN ----')

    args = utils.get_args()

    args.data_folder = '../../data'
    args.tensorboard_folder = '../../logs/core/'
    args.csv_folder = '../../data/csv/'
    args.model_folder = '../../logs/core/'

    t1 = time.time()
    input_len = [12]
    datasets = ['abilene', 'geant']
    model = 'gwn'
    te_algs = ['p0','p1', 'p2', 'ob']
    predict_len = [6]
    method = 'core'

    for dataset_id, dataset in enumerate(datasets):
        results = {'mlu': np.zeros(shape=(len(te_algs), len(predict_len))),
                   'rc': np.zeros(shape=(len(te_algs), len(predict_len)))}

        for te_alg_id, te_alg in enumerate(te_algs):
            for pred_len_id, pre_len in enumerate(predict_len):
                args.input_len = 12
                args.predict_len = pre_len
                args.dataset = dataset
                args.model = model
                args.method = method

                if args.predict_len >= 12 or args.dataset == 'germany':
                    args.train_batch_size = 64
                    args.val_batch_size = 64
                else:
                    args.train_batch_size = 128
                    args.val_batch_size = 128

                args = utils.args_adjust(args)
                args.test = True
                args.te_alg = te_alg
                args.timeout = 60
                _, mlu, rc = run_te(args)
                results['mlu'][te_alg_id, pred_len_id] = np.mean(mlu)
                results['rc'][te_alg_id, pred_len_id] = np.mean(rc)

        os.makedirs('../../results/core/exp9/', exist_ok=True)
        for k, v in results.items():
            np.savetxt(f'../../results/core/exp9/exp9_{dataset}_{k}.txt', results[k], delimiter=',')

    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', date.today())


exp_9()
