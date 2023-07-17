import os

import numpy as np
import pandas as pd
import torch

EPS = 1e-8


# Loss function
def mae_u(preds, labels, lamda=2.0):
    err = preds - labels
    err[err < 0.0] = err[err < 0.0] * lamda

    return torch.mean(torch.abs(err))


def mse_u(preds, labels, lamda=2.0):
    err = preds - labels
    err[err < 0.0] = err[err < 0.0] * lamda

    return torch.mean(err ** 2)


def rse(preds, labels):
    return torch.sum((preds - labels) ** 2) / torch.sum((labels + EPS) ** 2)


def mae(preds, labels):
    return torch.mean(torch.abs(preds - labels))


def mse(preds, labels):
    return torch.mean((preds - labels) ** 2)


def mape(preds, labels):
    return torch.mean(torch.abs((preds - labels) / (labels + EPS)))


def rmse(preds, labels):
    return torch.sqrt(torch.mean((preds - labels) ** 2))


def calc_metrics(preds, labels):
    return rse(preds, labels), mae(preds, labels), mse(preds, labels), mape(preds, labels), rmse(preds, labels)


def mape_np(X, M, W, Wo):
    X[X <= 0] = EPS
    sample_mape = np.sum(W * Wo * np.abs((X - M) / X)) / (np.sum(W * Wo) + EPS)
    infer_mape = np.sum((1 - W * Wo) * np.abs((X - M) / X)) / (np.sum(1 - W * Wo) + EPS)
    return float(sample_mape), float(infer_mape)


def rmse_np(X, M, W, Wo):
    sample_rmse = np.sqrt(np.sum(W * Wo * (X - M) ** 2) / (np.sum(W * Wo) + EPS))
    infer_rmse = np.sqrt(np.sum((1 - W * Wo) * (X - M) ** 2) / (np.sum(1.0 - W * Wo) + EPS))
    return float(sample_rmse), float(infer_rmse)


def mae_np(X, M, W, Wo):
    sample_mae = np.sum(W * Wo * np.abs(X - M)) / (np.sum(W * Wo) + EPS)
    infer_mae = np.sum((1 - W * Wo) * np.abs(X - M)) / (np.sum(1.0 - W * Wo) + EPS)
    return float(sample_mae), float(infer_mae)


# RSE numpy performance metric
def rse_np(X, M, W, Wo):
    sample_rse = np.sqrt(np.sum(W * Wo * (X - M) ** 2) / (np.sum(W * Wo * X ** 2) + EPS))
    infer_rse = np.sqrt(np.sum((1 - W * Wo) * (X - M) ** 2) / (np.sum((1 - W * Wo) * X ** 2) + EPS))

    return float(sample_rse), float(infer_rse)


# MSE numpy performance metric
def mse_np(X, M, W, Wo):
    sample_mse = np.sum(W * Wo * (X - M) ** 2) / (np.sum(W * Wo) + EPS)
    infer_mse = np.sum((1 - W * Wo) * (X - M) ** 2) / (np.sum(1.0 - W * Wo) + EPS)
    return float(sample_mse), float(infer_mse)


def analysing_results(y_cs, y_real, logger, args):
    test_met = []
    for i in range(y_cs.shape[1]):
        pred = y_cs[:, i, :]
        real = y_real[:, i, :]
        test_met.append([x.item() for x in calc_metrics(pred, real)])
    test_met_df = pd.DataFrame(test_met, columns=['rse', 'mae', 'mse', 'mape', 'rmse']).rename_axis('t')
    test_met_df.round(6).to_csv(os.path.join(logger.log_dir, 'summarized_test_metrics_cs_{}.csv'.format(args.cs)))
    print('Prediction Accuracy:')
    print(test_met_df)

    # Calculate metrics per cycle
    test_met = []
    for t in range(y_cs.shape[0]):
        for i in range(y_cs.shape[1]):
            pred = y_cs[t, i, :]
            real = y_real[t, i, :]
            test_met.append([x.item() for x in calc_metrics(pred, real)])
    test_met_df = pd.DataFrame(test_met, columns=['rse', 'mae', 'mse', 'mape', 'rmse']).rename_axis('t')
    test_met_df.round(6).to_csv(os.path.join(logger.log_dir, 'test_metrics_cs_{}.csv'.format(args.cs)))

    # Calculate metrics for top k% flows

    for tk in [1, 2, 3, 4, 5]:
        y_real = y_real.squeeze(dim=1)
        means = torch.mean(y_real, dim=0)
        top_idx = torch.argsort(means, descending=True)
        top_idx = top_idx[:int(tk * y_real.shape[1] / 100)]

        ycs_1 = y_cs[:, :, top_idx]
        y_real_1 = y_real[:, top_idx]

        test_met = []
        pred = ycs_1[:, 0, :]
        real = y_real_1
        test_met.append([x.item() for x in calc_metrics(pred, real)])
        test_met_df = pd.DataFrame(test_met, columns=['rse', 'mae', 'mse', 'mape', 'rmse']).rename_axis('t')
        test_met_df.round(6).to_csv(
            os.path.join(logger.log_dir, 'summarized_test_metrics_top1_cs_{}_tk_{}.csv'.format(args.cs, tk)))

        test_met = []
        for t in range(y_cs.shape[0]):
            pred = ycs_1[t, 0, :]
            real = y_real_1[t, :]
            test_met.append([x.item() for x in calc_metrics(pred, real)])
        test_met_df = pd.DataFrame(test_met, columns=['rse', 'mae', 'mse', 'mape', 'rmse']).rename_axis('t')
        test_met_df.round(6).to_csv(os.path.join(logger.log_dir, 'test_metrics_top1_cs_{}_tk_{}.csv'.format(
            args.cs, tk)))
