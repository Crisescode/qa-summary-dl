import numpy as np
from .config import train_x_path, train_y_path, test_x_path


def load_dataset():
    """
    :return:
    """
    train_X = np.loadtxt(train_x_path)
    train_Y = np.loadtxt(train_y_path)
    test_X = np.loadtxt(test_x_path)

    train_X.dtype = "float64"
    train_Y.dtype = "float64"
    test_X.dtype = "float64"

    return train_X, train_Y, test_X
