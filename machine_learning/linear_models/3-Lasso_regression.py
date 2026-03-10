#!/usr/bin/env python3
""" This module contains lasso_regression() """

from sklearn import linear_model


Lasso = linear_model.Lasso


def lasso_regression(random_state):
    """ Creates and returns a Lasso Regression model using Scikit-learn.
        > random_state .... used to shuffle the data for reproducibility
    """
    return Lasso(random_state=random_state)
