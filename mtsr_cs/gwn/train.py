import sys

sys.path.append('../../../')

import time
import warnings
from datetime import datetime

import torch
from tqdm import trange

import models
import utils
from mtsr.mtsr_cs.ksvd import KSVD
from mtsr.mtsr_cs.pursuit import sparse_coding
from mtsr.routing import *

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


def get_psi(args, samples=2000):
    X = utils.load_raw(args)
    train, val, test_list = utils.train_test_split(X, args.dataset)
    X = train[-samples:]

    X_temp = np.array([np.max(X[args.seq_len_x + i:
                                args.seq_len_x + i + args.seq_len_y], axis=0) for i in
                       range(samples - args.seq_len_x - args.seq_len_y)])

    X_temp_max = np.max(X_temp, axis=1)
    X_temp = X_temp[X_temp_max > 1.0]

    N_F = X.shape[1]
    D = np.zeros(shape=(N_F, N_F))

    psiT, ST = KSVD(D).fit(X_temp)
    return psiT, ST


def get_phi(top_k_index, nseries):
    G = np.zeros((top_k_index.shape[0], nseries))

    for i, j in enumerate(G):
        j[top_k_index[i]] = 1

    return G


def main(args, **model_kwargs):
    device = torch.device(args.device)
    args.device = device
    if 'abilene' in args.dataset:
        args.nNodes = 12
        args.day_size = 288
    elif 'geant' in args.dataset:
        args.nNodes = 22
        args.day_size = 96
    elif 'brain' in args.dataset:
        args.nNodes = 9
        args.day_size = 1440
    elif 'sinet' in args.dataset:
        args.nNodes = 74
        args.day_size = 288
    elif 'renater' in args.dataset:
        args.nNodes = 30
        args.day_size = 288
    elif 'surfnet' in args.dataset:
        args.nNodes = 50
        args.day_size = 288
    elif 'uninett' in args.dataset:
        args.nNodes = 74
        args.day_size = 288
    else:
        raise ValueError('Dataset not found!')

    train_loader, val_loader, test_loader, total_timesteps, total_series = utils.get_dataloader(args)
    args.nSeries = int(args.mon_rate * total_series / 100)

    in_dim = 1
    if args.tod:
        in_dim += 1
    if args.ma:
        in_dim += 1
    if args.mx:
        in_dim += 1

    args.in_dim = in_dim

    aptinit, supports = utils.make_graph_inputs(args, device)

    model = models.GWNet.from_args(args, supports, aptinit, **model_kwargs)
    model.to(device)
    logger = utils.Logger(args)

    engine = utils.Trainer.from_args(model=model, scaler=None,
                                     scaler_top_k=test_loader.dataset.scaler_topk, args=args)

    utils.print_args(args)

    if not args.test:
        iterator = trange(args.epochs)

        try:
            if os.path.isfile(logger.best_model_save_path):
                print('Model checkpoint exist!')
                print('Load model checkpoint? (y/Y/Yes/yes/)')
                _in = input()
                if _in == 'y' or _in == 'yes' or _in == 'Y' or _in == 'Yes':
                    print('Loading model...')
                    engine.model.load_state_dict(torch.load(logger.best_model_save_path))
                else:
                    print('Training new model')

            for epoch in iterator:
                train_loss, train_rse, train_mae, train_mse, train_mape, train_rmse = [], [], [], [], [], []
                for iter, batch in enumerate(train_loader):
                    # x = batch['x']  # [b, seq_x, n, f]
                    # y = batch['y']  # [b, seq_y, n]
                    # sys.exit()
                    x = batch['x_top_k']
                    y = batch['y_top_k']

                    if y.max() == 0:
                        continue
                    loss, rse, mae, mse, mape, rmse = engine.train(x, y)
                    train_loss.append(loss)
                    train_rse.append(rse)
                    train_mae.append(mae)
                    train_mse.append(mse)
                    train_mape.append(mape)
                    train_rmse.append(rmse)

                engine.scheduler.step()
                with torch.no_grad():
                    val_loss, val_rse, val_mae, val_mse, val_mape, val_rmse = engine.eval(val_loader)
                m = dict(train_loss=np.mean(train_loss), train_rse=np.mean(train_rse),
                         train_mae=np.mean(train_mae), train_mse=np.mean(train_mse),
                         train_mape=np.mean(train_mape), train_rmse=np.mean(train_rmse),
                         val_loss=np.mean(val_loss), val_rse=np.mean(val_rse),
                         val_mae=np.mean(val_mae), val_mse=np.mean(val_mse),
                         val_mape=np.mean(val_mape), val_rmse=np.mean(val_rmse))

                description = logger.summary(m, engine.model, epoch=epoch)

                if logger.stop:
                    break

                description = 'Epoch: {} '.format(epoch) + description
                iterator.set_description(description)
        except KeyboardInterrupt:
            pass
    else:
        # Metrics on test dataset
        engine.model.load_state_dict(torch.load(logger.best_model_save_path))
        with torch.no_grad():
            test_met_df, x_gt, y_gt, yhat, y_real, y_real_top_k = engine.test(test_loader, engine.model,
                                                                              args.out_seq_len)
            test_met_df.round(6).to_csv(os.path.join(logger.log_dir, 'test_metrics.csv'))
            # print('Prediction Accuracy:')
            # print(utils.summary(logger.log_dir))

        x_gt = x_gt.cpu().data.numpy()  # [timestep, seq_x, seq_y]
        y_gt = y_gt.cpu().data.numpy()
        yhat = yhat.cpu().data.numpy()
        np.save(os.path.join(logger.log_dir, 'yhat'), yhat)  # saved yhat

        top_k_index = test_loader.dataset.Topkindex
        ygt_shape = y_gt.shape
        if args.cs:
            print('|--- Traffic reconstruction using CS')

            # obtain psi, G, R
            psi_save_path = os.path.join(args.datapath, 'cs/saved_psi/')
            if not os.path.exists(psi_save_path):
                os.makedirs(psi_save_path)
            psi_save_path = os.path.join(psi_save_path, '{}_{}_{}_psi.pkl'.format(args.dataset,
                                                                                  args.seq_len_x,
                                                                                  args.seq_len_y))
            if not os.path.isfile(psi_save_path):
                print('|--- Calculating psi, phi')
                psiT, ST = get_psi(args)
                obj = {
                    'psiT': psiT,
                    'ST': ST
                }
                with open(psi_save_path, 'wb') as fp:
                    pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)
                    fp.close()
            else:
                print('|--- Loading psi, phi from {}'.format(psi_save_path))

                with open(psi_save_path, 'rb') as fp:
                    obj = pickle.load(fp)
                    fp.close()
                psiT = obj['psiT']

            phi = get_phi(top_k_index, total_series)
            print('psiT: ', psiT.shape)
            print('phi: ', phi.shape)

            yhat = np.squeeze(yhat, axis=1)  # shape(n, k)
            ShatT = sparse_coding(ZT=yhat, phiT=phi.T, psiT=psiT)
            y_cs = np.dot(ShatT, psiT)
            y_cs[:, top_k_index] = yhat
            y_cs = np.expand_dims(y_cs, axis=1)  # shape(n, 1, N_F)
        else:
            print('|--- No traffic reconstruction')
            y_cs = np.zeros(shape=(ygt_shape[0], 1, ygt_shape[-1]))
            y_cs[:, :, top_k_index] = yhat

        x_gt = torch.from_numpy(x_gt).to(args.device)
        y_gt = torch.from_numpy(y_gt).to(args.device)
        y_cs = torch.from_numpy(y_cs).to(args.device)
        y_cs[y_cs <= 0.0] = 0.0

        # Calculate error
        utils.analysing_results(y_cs, y_real, logger, args)

        # run traffic engineering
        x_gt = x_gt.cpu().data.numpy()  # [timestep, seq_x, seq_y]
        y_gt = y_gt.cpu().data.numpy()
        y_cs = y_cs.cpu().data.numpy()
        y_real = y_real.cpu().data.numpy()

        np.save(os.path.join(logger.log_dir, 'x_gt'), x_gt)
        np.save(os.path.join(logger.log_dir, 'y_gt'), y_gt)
        np.save(os.path.join(logger.log_dir, 'y_cs'), y_cs)
        np.save(os.path.join(logger.log_dir, 'y_real'), y_real)
        np.save(os.path.join(logger.log_dir, 'y_hat'), yhat)

        print('\n{} mon_rate:{} cs: {}'.format(args.dataset, args.mon_rate, args.cs))
        if args.run_te != 'None':
            print('x_gt ', x_gt.shape)
            print('y_gt ', y_gt.shape)
            print('y_cs ', y_cs.shape)

            x_gt[x_gt <= 0] = 0.0
            y_gt[y_gt <= 0] = 0.0
            y_cs[y_cs <= 0] = 0.0
            run_te(x_gt, y_gt, y_cs, logger, args)

        print('\n            ----------------------------\n')


if __name__ == "__main__":
    args = utils.get_args()
    t1 = time.time()
    main(args)
    t2 = time.time()
    mins = (t2 - t1) / 60
    print("Total time spent: {:.2f} seconds".format(mins))
    print('Date&Time: ', datetime.now())
