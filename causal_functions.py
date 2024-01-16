import os
import time 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nonlincausality as nlc 
from nonlincausality.nonlincausality import nonlincausalityARIMA, nonlincausalityMLP

from statsmodels.tsa.stattools import grangercausalitytests

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
plot = False 


def get_granger_linear(X, Y, lag):
    data = np.array([X, Y]).T.astype(np.float64)
    results = grangercausalitytests(data, maxlag=[lag])
    ssr_ftest = results[lag][0]["ssr_ftest"]
    chi_test = results[lag][0]["ssr_chi2test"]
    return [("F-statistic", ssr_ftest[0]), ("p-value", ssr_ftest[1]),
            ("chi2", chi_test[0]), ("p-value", chi_test[1])]


def get_granger_arima(X, Y, lag):
    data = np.array([X, Y]).T.astype(np.float64)
    results = nonlincausalityARIMA(
        x=data, 
        maxlag=[lag],
        x_test=data)

    p_value = results[lag].p_value
    test_statistic = results[lag]._test_statistic

    best_errors_X = results[lag].best_errors_X
    best_errors_XY = results[lag].best_errors_XY
    cohens_d = np.abs(
        (np.mean(np.abs(best_errors_X)) - np.mean(np.abs(best_errors_XY)))
        / np.std([best_errors_X, best_errors_XY])
    )
    return [
        ("arima_pval", p_value),
        ("arima_stat", test_statistic),
        ("arima_cohen", cohens_d),
    ]

def get_granger_mlp(X, Y, lag):
    nlen = len(X)
    data = np.array([X, Y]).T.astype(np.float64)
    data_train = data[:int(nlen*0.8)]
    data_test = data[int(nlen*0.8):]
    results = nonlincausalityMLP(
        x=data_train, 
        maxlag=[lag],
        Dense_layers=2,
        Dense_neurons=[100, 100],
        x_test=data_test,
        run=1,
        epochs_num=50,
        batch_size_num=128,
        verbose=True,
        plot=True,
    )

    p_value = results[lag].p_value
    test_statistic = results[lag]._test_statistic

    best_errors_X = results[lag].best_errors_X
    best_errors_XY = results[lag].best_errors_XY
    cohens_d = np.abs(
        (np.mean(np.abs(best_errors_X)) - np.mean(np.abs(best_errors_XY)))
        / np.std([best_errors_X, best_errors_XY])
    )
    return [
        ("mlp_pval", p_value),
        ("mlp_stat", test_statistic),
        ("mlp_cohen", cohens_d),
    ]

def data_generator(len=10000):
    np.random.seed(10)
    y = (
        np.cos(np.linspace(0, 20, 10_100))
        + np.sin(np.linspace(0, 3, 10_100))
        - 0.2 * np.random.random(10_100)
    )
    np.random.seed(20)
    x = 2 * y ** 3 - 5 * y ** 2 + 0.3 * y + 2 - 0.05 * np.random.random(10_100)
    return x[:len], y[:len]

## run 
lag = 10
x, y = data_generator(len=10000)
if plot:
    fig, ax = plt.subplots(2, 2, figsize=(12, 4))
    ax[0, 0].plot(x)
    ax[0, 0].plot(y)
    ax[0, 0].set_title("Original Data")


linear_gc = get_granger_linear(x, y, lag)
arima_gc = get_granger_arima(x, y, lag)
mlp_gc = get_granger_mlp(x, y, lag)

print(f"Linear Granger Causality y-causing-X: {linear_gc}")
print(f"ARIMA Granger Causality y-causing-X: {arima_gc}")
print(f"MLP Granger Causality y-causing-X: {mlp_gc}")

linear_gc = get_granger_linear(y, x, lag)
# arima_gc = get_granger_arima(y, x, lag)
mlp_gc = get_granger_mlp(y, x, lag)

print(f"Linear Granger Causality x-causing-y: {linear_gc}")
print(f"ARIMA Granger Causality x-causing-y: {arima_gc}")
print(f"MLP Granger Causality x-causing-y: {mlp_gc}")
