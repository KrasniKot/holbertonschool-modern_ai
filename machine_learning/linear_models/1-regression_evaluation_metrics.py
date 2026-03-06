#!/usr/bin/env python3

""" This module contains the function: evaluation_metrics_for_regression """

from sklearn import metrics

mse = metrics.mean_squared_error
rmse = metrics.root_mean_squared_error
mae = metrics.mean_absolute_error
r2 = metrics.r2_score


def evaluation_metrics_for_regression(y, yhat):
    """ Computes common evaluation metrics for regression tasks
        > y ...... A 1D NumPy array containing the true target values.
        > yhat ... A 1D NumPy array containing the predicted target values.

        >>> A tuple containing (mse, rmse, mae, r2)
    """
    return mse(y, yhat), rmse(y, yhat), mae(y, yhat), r2(y, yhat)
