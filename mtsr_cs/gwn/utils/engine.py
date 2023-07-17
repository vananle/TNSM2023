import torch.optim as optim

from .metric import *


class Trainer():
    def __init__(self, model, scaler, scaler_top_k, lrate, wdecay, clip=3, lr_decay_rate=.97, lossfn='mae',
                 verbose=False):
        self.model = model

        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scaler = scaler
        self.scaler_top_k = scaler_top_k
        self.clip = clip
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: lr_decay_rate ** epoch)

        if lossfn == 'mae':
            self.lossfn = mae
        elif lossfn == 'mse':
            self.lossfn = mse
        elif lossfn == 'mae_u':
            self.lossfn = mae_u
        elif lossfn == 'mse_u':
            self.lossfn = mse_u
        else:
            raise ValueError('Loss fn not found!')

        self.verbose = verbose

    @classmethod
    def from_args(cls, model, scaler, scaler_top_k, args):
        return cls(model, scaler, scaler_top_k, args.learning_rate, args.weight_decay, clip=args.clip,
                   lr_decay_rate=args.lr_decay_rate, lossfn=args.loss_fn, verbose=args.verbose)

    def train(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        # input = torch.nn.functional.pad(input, (1, 0, 0, 0))

        output = self.model(x)  # now, output = [bs, seq_y, n]
        if self.scaler_top_k is not None:
            output = self.scaler_top_k.inverse_transform(output)

        if self.verbose:
            print('x: ', x.shape)
            print('y: ', y.shape)
            print('out: ', output.shape)

        loss = self.lossfn(output, y)
        rse, mae, mse, mape, rmse = calc_metrics(output, y)
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        return loss.item(), rse.item(), mae.item(), mse.item(), mape.item(), rmse.item()

    def _eval(self, input, real_val):
        self.model.eval()

        output = self.model(input)  # now, output = [bs, seq_y, n]

        if self.scaler_top_k is not None:
            output = self.scaler_top_k.inverse_transform(output)

        output = torch.clamp(output, min=0., max=10e10)
        loss = self.lossfn(output, real_val)
        rse, mae, mse, mape, rmse = calc_metrics(output, real_val)

        return loss.item(), rse.item(), mae.item(), mse.item(), mape.item(), rmse.item()

    def test(self, test_loader, model, out_seq_len):
        model.eval()
        outputs = []
        y_real_top_k = []
        x_gt = []
        y_gt = []
        y_real = []
        for _, batch in enumerate(test_loader):

            x_top_k = batch['x_top_k']
            y_top_k = batch['y_top_k']

            preds_top_k = model(x_top_k)
            if self.scaler_top_k is not None:
                preds_top_k = self.scaler_top_k.inverse_transform(preds_top_k)

            outputs.append(preds_top_k)
            y_real_top_k.append(y_top_k)

            x_gt.append(batch['x_gt'])
            y_gt.append(batch['y_gt'])
            y_real.append(batch['y_real'])

        yhat = torch.cat(outputs, dim=0)
        y_real_top_k = torch.cat(y_real_top_k, dim=0)
        x_gt = torch.cat(x_gt, dim=0)
        y_gt = torch.cat(y_gt, dim=0)
        y_real = torch.cat(y_real, dim=0)
        test_met = []

        yhat[yhat < 0.0] = 0.0

        for i in range(out_seq_len):
            pred = yhat[:, i, :]
            pred = torch.clamp(pred, min=0., max=10e10)
            real = y_real_top_k[:, i, :]
            test_met.append([x.item() for x in calc_metrics(pred, real)])
        test_met_df = pd.DataFrame(test_met, columns=['rse', 'mae', 'mse', 'mape', 'rmse']).rename_axis('t')
        return test_met_df, x_gt, y_gt, yhat, y_real, y_real_top_k

    def eval(self, val_loader):
        """Run validation."""
        val_loss, val_rse, val_mae, val_mse, val_mape, val_rmse = [], [], [], [], [], []
        for _, batch in enumerate(val_loader):
            # x = batch['x']  # [b, seq_x, n, f]
            # y = batch['y']  # [b, seq_y, n]

            x = batch['x_top_k']
            y = batch['y_top_k']

            metrics = self._eval(x, y)
            val_loss.append(metrics[0])
            val_rse.append(metrics[1])
            val_mae.append(metrics[2])
            val_mse.append(metrics[3])
            val_mape.append(metrics[4])
            val_rmse.append(metrics[5])

        return val_loss, val_rse, val_mae, val_mse, val_mape, val_rmse
