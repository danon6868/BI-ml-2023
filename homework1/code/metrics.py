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

    true_pos = y_pred[(y_true == y_pred) & (y_pred == 1)].shape[0]
    false_pos = y_pred[(y_true != y_pred) & (y_pred == 1)].shape[0]
    true_neg = y_pred[(y_true == y_pred) & (y_pred == 0)].shape[0]
    false_neg = y_pred[(y_true != y_pred) & (y_pred == 0)].shape[0]

    precision = true_pos/(true_pos + false_pos)
    recall = true_pos/(true_pos + false_neg)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / (true_pos + false_pos + true_neg + false_neg)

    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    true_pos = y_pred[(y_true == y_pred) & (y_pred == 1)].shape[0]
    false_pos = y_pred[(y_true != y_pred) & (y_pred == 1)].shape[0]
    true_neg = y_pred[(y_true == y_pred) & (y_pred == 0)].shape[0]
    false_neg = y_pred[(y_true != y_pred) & (y_pred == 0)].shape[0]

    return (true_pos + true_neg)/(true_pos + true_neg + false_pos + false_neg)

def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    return 1 - ((y_pred - y_true)**2).sum() / ((y_true - y_true.mean())**2).sum()


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    return np.mean((y_pred - y_true)**2)


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    return (np.abs(y_true - y_pred)).sum() / y_pred.shape[0]
    # MAE = sum of absolute errors divided by the sample size