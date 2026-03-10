#!/usr/bin/env python3
""" This module contains ridge_regression() """

from sklearn.linear_model import Ridge


def ridge_regression(random_state):
    """ Creates and returns a Ridge Regression model using Scikit-learn.
        > random_state .... used to shuffle the data for reproducibility
    """
    return Ridge(random_state=random_state)
