#!/usr/bin/env python3
""" This module contains get_shap_explainer_and_values """

import shap


def get_shap_explainer_and_values(model, X_train, X_test):
    """ Generates model explanations using the SHAP library
        > model ...... a trained regression model
        > x_train .... input data to initialise the explainer
        > x_test ..... input data to explain
    """
    explainer = shap.LinearExplainer(model, X_train)

    return explainer, explainer(X_test)
