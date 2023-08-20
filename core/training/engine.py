import copy
import sys
import time

import numpy as np
import torch.nn

sys.path.append('../')

from core.utils import *
from core.compressive_sensing.utils import get_phi, get_psi
from core.compressive_sensing.pursuit import sparse_coding

# Building Model
lossfn = torch.nn.MSELoss()


class TrainEngine:
    def __init__(self, data, model, args):
        self.args = args

        self.lr = args.lr

        self.model = model
        self.train_loader = data['train_loader']
        self.val_loader = data['val_loader']
        self.test_loader = data['test_loader']
        self.scaler = data['scaler']

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=1, gamma=0.75)
        self.num_epochs = args.num_epochs
        self.monitor = Monitor(args, self.num_epochs)
        self.info = {}

        self.best_metric = np.inf
        self.num_not_improved = 0
        self.metric = 'mse'

        # store data
        self.data = data

    def train(self):
        self.best_metric = np.inf
        self.num_not_improved = 0

        for epoch in range(self.num_epochs):
            self.info = {}
            _metrics = self.training(mode='train')
            self.add_info(_metrics, mode='train')
            with torch.no_grad():
                _metrics = self.training(mode='val')
                self.add_info(_metrics, mode='val')

                _metrics = self.training(mode='test')
                self.add_info(_metrics, mode='test')

            stop = self.check_stopping_condition()
            self.monitor.step(self.info)

            if stop:
                break
            if (self.num_not_improved >= int(self.args.patience / 2)) and epoch % 10 == 0:
                self.scheduler.step()

    def test(self):
        self.load_model()
        with torch.no_grad():
            _metrics = self.training(mode='test', scaler=True)

        return _metrics

    def tm_reconstruction(self):

        y_gt = self.data['test/y_gt']
        y_hat = self.data['test/y_hat']

        mon_index = self.data['test/mon_index']
        if self.args.mon_per < 1.0:
            if self.args.method == 'mtsr_cs':

                print('|--- Traffic reconstruction using CS')

                # obtain psi, G, R
                psi_save_path = os.path.join(self.args.data_folder, 'cs/')
                if not os.path.exists(psi_save_path):
                    os.makedirs(psi_save_path)
                psi_save_path = os.path.join(psi_save_path, '{}_{}_{}_psi.npz'.format(self.args.dataset,
                                                                                      self.args.input_len,
                                                                                      self.args.predict_len))
                if not os.path.isfile(psi_save_path):
                    print('|--- Calculating psi, phi')
                    psiT, ST = get_psi(args=self.args, data=self.data)
                    obj = {
                        'psiT': psiT,
                        'ST': ST
                    }
                    np.savez_compressed(psi_save_path, **obj)

                else:
                    print('|--- Loading psi, phi from {}'.format(psi_save_path))

                    obj = np.load(psi_save_path)
                    psiT = obj['psiT']

                y_cs = np.zeros(shape=(y_hat.shape[0], self.args.num_flow))
                run_time = []

                for t in range(y_hat.shape[0]):
                    time_s = time.time()

                    mon_index_t = mon_index[t]

                    phi = get_phi(mon_index_t, self.args.num_flow)

                    # yhat_i = np.squeeze(y_hat, axis=1)  # shape(n, k)
                    y_hat_t = y_hat[t]  # shape(1, k)
                    ShatT = sparse_coding(ZT=y_hat_t, phiT=phi.T, psiT=psiT)
                    ycs = np.dot(ShatT, psiT)
                    ycs[:, mon_index_t] = y_hat_t
                    y_cs[t] = ycs.flatten()  # shape(n, N_F)
                    run_time.append(time.time() - time_s)

                run_time = np.array(run_time)
                np.save(f'/home/anle/mtsr-cs-runtime-{self.args.dataset}-{self.args.mon_per}.npy', run_time)

            elif self.args.method == 'mtsr_nocs':
                y_cs = np.zeros(shape=(y_hat.shape[0], self.args.num_flow))
                for t in range(y_hat.shape[0]):
                    mon_index_t = mon_index[t]
                    y_hat_t = y_hat[t]  # shape(1, k)
                    y_cs[t, mon_index_t] = y_hat_t  # shape(n, N_F)
            else:
                raise NotImplementedError
        else:
            y_cs = np.zeros(shape=(y_hat.shape[0], self.args.num_flow))
            for t in range(y_hat.shape[0]):
                mon_index_t = mon_index[t]
                y_hat_t = y_hat[t]  # shape(1, k)
                y_cs[t, mon_index_t] = y_hat_t  # shape(n, N_F)

        rse, mae, mse, mape, rmse = calc_metrics_np(y_cs, self.data['test/y_gt_max'])

        self.data.update({
            'test/y_cs': y_cs,
            'mae_y_cs': mae
        })

    def reshape_input(self, x):
        if self.args.model == 'gwn':
            if len(x.size()) < 4:
                x = torch.unsqueeze(x, dim=-1)
        else:
            x = x
        return x

    def training(self, scaler=False, mode='train'):

        model = self.model
        optimizer = self.optimizer
        loader = self.get_loader(mode)

        if mode == 'train':
            model.train()
        else:
            model.eval()

        y_hat = []
        batch_loss, batch_rse, batch_mae, batch_mse, batch_mape, batch_rmse = [], [], [], [], [], []
        for idx, (x, y) in enumerate(loader):
            if y.max() == 0:
                continue
            if mode == 'train':
                optimizer.zero_grad()

            output = model(self.reshape_input(x))  # now, output = [bs, seq_y, n]

            if mode == 'train':
                y_hat.append(copy.deepcopy(output.detach()))
            else:
                y_hat.append(copy.deepcopy(output))

            if len(output.size()) != len(y.size()):
                output = torch.reshape(output, y.shape)

            loss = lossfn(output, y)
            rse, mae, mse, mape, rmse = calc_metrics(output, y)

            if mode == 'train':
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
                optimizer.step()

            batch_loss.append(loss.item())
            batch_rse.append(rse.item())
            batch_mae.append(mae.item())
            batch_mse.append(mse.item())
            batch_mape.append(mape.item())
            batch_rmse.append(rmse.item())

        y_hat = torch.cat(y_hat, dim=0)
        y_hat = y_hat.cpu().numpy()

        if scaler:
            y_hat_shape = y_hat.shape
            if len(y_hat_shape) > 2:
                y_hat = np.reshape(y_hat, newshape=(y_hat_shape[0], -1))
                y_hat = self.scaler.inverse_transform(y_hat)
                y_hat = np.reshape(y_hat, newshape=y_hat_shape)
            else:
                y_hat = self.scaler.inverse_transform(y_hat)

        self.data.update({
            f'{mode}/y_hat': y_hat
        })

        _metrics = {'loss': sum(batch_loss) / len(batch_loss),
                    'rse': sum(batch_rse) / len(batch_rse),
                    'mae': sum(batch_mae) / len(batch_mae),
                    'mse': sum(batch_mse) / len(batch_mse),
                    'mape': sum(batch_mape) / len(batch_mape),
                    'rmse': sum(batch_rmse) / len(batch_rmse)}

        return _metrics

    def check_stopping_condition(self):
        # stopping condition
        metric = np.mean(self.info[f'val/{self.metric}'])
        if metric < self.best_metric:  # improving metric (MSE or MAE)
            self.best_metric = metric
            self.num_not_improved = 0
            self.save_model()
        else:
            self.num_not_improved += 1

        if self.num_not_improved >= self.args.patience:
            return True
        return False

    def load_model(self):
        self.model = self.monitor.load_model(self.model, tag=None)

    def save_model(self):
        self.monitor.save_model(self.model)

    def add_info(self, metrics, mode):
        for key in metrics:
            info_key = f'{mode}/{key}'
            if info_key not in self.info:
                self.info[info_key] = []
            self.info[info_key].append(metrics[key])

    def get_loader(self, mode='train'):
        if mode == 'train':
            return self.train_loader
        elif mode == 'val':
            return self.val_loader
        elif mode == 'test':
            return self.test_loader
        else:
            raise ValueError

    def save_data(self, data):
        self.monitor.save_data(data)

    def load_data(self):
        data = self.monitor.load_data()
        return data
