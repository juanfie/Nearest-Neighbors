import numpy as np


def sub_mape(x, y):
    return np.abs((x - y) / x)


def mape(a, f):
    """Calculates the Mean average percentage error between two sets a (actual) and f (forecast)"""
    c = list(a)
    d = list(f)
    indexes = []
    for i in range(len(a)):
        if c[i] == 0 and d[i] == 0:
            indexes.append(i)
    c = np.array([x for i, x in enumerate(a) if i not in indexes])
    d = np.array([x for i, x in enumerate(f) if i not in indexes])
    s = 0
    for x, y, in zip(c, d):
        if x != 0:
            s += sub_mape(x, y)
    return s / float(c.shape[0]) * 100
    # return np.mean(np.abs(d - c) / c) * 100


def sub_smape(x, y):
    return np.abs(x - y) / ((np.abs(x) + np.abs(y)) / 2)


def smape(a, f):
    """Calculates the symmetric mean average percentage error between two sets a (actual) and f (forecast)"""
    c = list(a)
    d = list(f)
    indexes = []
    for i in range(len(a)):
        if c[i] == 0 and d[i] == 0:
            indexes.append(i)
    c = np.array([x for i, x in enumerate(a) if i not in indexes])
    d = np.array([x for i, x in enumerate(f) if i not in indexes])
    return np.mean(np.abs(d - c) / ((np.abs(c) + np.abs(d)) / 2)) * 100


def sub_mse(x, y):
    return np.power((x - y), 2)


def mse(a, f):
    """Calculates the Mean Squared Error between two sets a (actual) and f (forecast)"""
    a = np.array(a)
    f = np.array(f)
    return np.mean(np.power((a - f), 2))


def nmse(a, f):
    """Normalized Mean Squared error between two sets a (actual) and f (forecast)"""
    a = np.array(a)
    f = np.array(f)
    return np.mean(np.power((a - f), 2) / (np.mean(a) + np.mean(f)))


def mae(forecast, validation):
    e = -1
    if len(forecast) == len(validation):
        f = np.array(forecast)
        v = np.array(validation)
        e = np.mean(np.abs(v - f))
    else:
        raise ValueError('length of forecast is not equal to length of validation')
    return e