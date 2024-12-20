import numpy as np


def load_data_code():
    X = np.load("E:\\gitProject\\Machine-Learning-Specialization\\_resources_ML_spec\\C2_W1\\data\\X.npy")
    y = np.load("E:\\gitProject\\Machine-Learning-Specialization\\_resources_ML_spec\\C2_W1\\data\\y.npy")
    X = X[0:1000]
    y = y[0:1000]
    return X, y


def load_data():
    X = np.load("../_resources_ML_spec/C2_W1/data/X.npy")
    y = np.load("../_resources_ML_spec/C2_W1/data/y.npy")
    X = X[0:1000]
    y = y[0:1000]
    return X, y


def load_weights():
    w1 = np.load("../_resources_ML_spec/C2_W1/data/w1.npy")
    b1 = np.load("../_resources_ML_spec/C2_W1/data/b1.npy")
    w2 = np.load("../_resources_ML_spec/C2_W1/data/w2.npy")
    b2 = np.load("../_resources_ML_spec/C2_W1/data/b2.npy")
    return w1, b1, w2, b2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
