import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    tp_n = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    tn_n = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    fp_n = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    fn_n = np.sum(np.logical_and(y_pred == 0, y_true == 1))

    precision = tp_n / (tp_n + fp_n)
    recall = tp_n / (tp_n + fn_n)

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0

    accuracy = (tp_n + tn_n) / (tp_n + tn_n + fn_n + fp_n)

    print(f'Accuracy = {accuracy}\nprecision = {precision}\nrecall = {recall}\nf1 = {f1}')


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    tp = (y_true == y_pred).sum()
    accuracy = tp / len(y_true)

    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2

    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = ((y_pred - y_true) ** 2).sum() / len(y_true)

    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mae = (y_pred - y_true).abs().sum() / len(y_true)

    return mae
