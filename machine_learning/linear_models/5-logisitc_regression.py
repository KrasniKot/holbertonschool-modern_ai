#!/usr/bin/env python3
""" This module contains Logistic_Regression_Model """

from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """ Creates a logistic regression model using Scikit-learn
        > random_state ..... An integer used to set the random seed for reproducibility
        
        >>> An untrained LogisticRegression instance.
    """
    return linear_model.LogisticRegression(random_state=random_state)
