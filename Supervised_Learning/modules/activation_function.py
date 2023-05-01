import numpy as np


def sigmoid():
    return (__eval_sigmoid, __eval_sigmoid_diff)


def __eval_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def __eval_sigmoid_diff(x):
    return __eval_sigmoid(x) * (1.0 - __eval_sigmoid(x))
