import numpy as np
import torch

EPS = 1e-8


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


def rse_np(preds, labels):
    return np.sum((preds - labels) ** 2) / np.sum((labels + EPS) ** 2)


def mae_np(preds, labels):
    return np.mean(np.abs(preds - labels))


def mse_np(preds, labels):
    return np.mean((preds - labels) ** 2)


def mape_np(preds, labels):
    return np.mean(np.abs((preds - labels) / (labels + EPS)))


def rmse_np(preds, labels):
    return np.sqrt(np.mean((preds - labels) ** 2))


def calc_metrics_np(preds, labels):
    return rse_np(preds, labels), mae_np(preds, labels), mse_np(preds, labels), \
           mape_np(preds, labels), rmse_np(preds, labels)
