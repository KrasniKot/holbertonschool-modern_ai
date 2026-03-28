#!/usr/bin/env python3
""" This module contains get_SVM_model """

from sklearn.svm import SVC


def get_SVM_model(name, random_state):
    """ Returns an SVM model with the specified kernel.
        > name ............ Type of kernel ('linear', 'poly', 'rbf')
        > random_state .... Seed for reproducibility

        >>> Configured SVM model
    """

    if name in ('linear', 'poly', 'rbf'):
        return SVC(kernel=name, random_state=random_state)

    error = "Invalid model name. Choose from 'linear', 'poly', or 'rbf'."
    raise ValueError(error)
