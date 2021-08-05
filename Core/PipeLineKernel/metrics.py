import sys
sys.path.append('..')
import numpy as np
import math as mt
from sklearn.metrics import mean_squared_error


y_orig = np.array([[1, 2, 3],
                    [1, 2, 3]])
y_pred = np.array([[1, 1, 0],
                   [1, 1, 0]])

def mse(data_original, data_predicted):
    """
    Calculate the root mean square error for our model
    formule: mse = sum(data_original_ui - data_predicted_ui) / len(data_original)

    Parameters
    ----------
    data_original : (2D ndarray)
        this is the matrix of know ratings
    data_predicted: (2D ndarray)
        this is the matrix predicted after training
    Returns
    -------
    mse : (float32)
    """
    data_predicted = data_predicted[data_original.nonzero()].flatten()
    data_original = data_original[data_original.nonzero()].flatten()
    return np.square(data_original - data_predicted).sum() / data_predicted.size

def rmse(data_original, data_predicted):
    """
    Calculate the root mean square error for our model
    formule: rmse = sqrt(sum(data_original_ui - data_predicted_ui) / len(data_original))

    Parameters
    ----------
    data_original : (2D ndarray)
        this is the matrix of know ratings
    data_predicted: (2D ndarray)
        this is the matrix predicted after training
    Returns
    -------
    rmse : (float32)
    """
    data_predicted = data_predicted[data_original.nonzero()].flatten()
    data_original = data_original[data_original.nonzero()].flatten()
    return mt.sqrt(np.square(data_original - data_predicted).sum() / data_original.size)

def mae(data_original, data_predicted):
    """
    Calculate median absolute error for our model
    formule : mae  = median(|data_original - data_predicted|)

    Parameters
    ----------
    data_original : (2D ndarray)
        this is the matrix of know ratings
    data_predicted: (2D ndarray)
        this is the matrix predicted after training
    Returns
    -------
    mae : (float)
    """
    data_predicted = data_predicted[data_original.nonzero()].flatten()
    data_original = data_original[data_original.nonzero()].flatten()
    return np.median(data_original - data_predicted)

def frobenuis(mat_init, mat_approx):
    error = 0
    for i in mat_approx.index.values:
        for v in mat_approx.columns.values:
            error += (mat_init.loc[i, v] - mat_approx.loc[i, v])**2
    return mt.sqrt(error)
# print(mt.sqrt(0.523209092297487))
