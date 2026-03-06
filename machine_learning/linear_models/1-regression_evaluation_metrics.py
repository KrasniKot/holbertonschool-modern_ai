#!/usr/bin/env python3

""" This module contains the function: evaluation_metrics_for_regression """

from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def evaluation_metrics_for_regression(y, yhat):
    """ Computes common evaluation metrics for regression tasks
        > y ...... A 1D NumPy array containing the true target values.
        > yhat ... A 1D NumPy array containing the predicted target values.

        >>> A tuple containing (mse, rmse, mae, r2)
    """
    metrics = {
        0: lambda y, yhat: mean_squared_error(y, yhat),
        1: lambda y, yhat: root_mean_squared_error(y, yhat),
        2: lambda y, yhat: mean_absolute_error(y, yhat),
        3: lambda y, yhat: r2_score(y, yhat)
    }

    # 0: mse, 1: rmse, 2: mae, 3: r2
    return tuple(metrics[i](y, yhat) for i in range(4))
