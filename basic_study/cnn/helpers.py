import numpy as np


def softmax(x):
    x_shifted = x - np.max(x, axis=0, keepdims=True)
    exps = np.exp(x_shifted)
    return exps / np.sum(exps, axis=0, keepdims=True)

def l2_regularization(W, reg_strength):
    loss = np.sum(np.square(W)) * reg_strength

    grad = 2 * reg_strength * W

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    probs = softmax(predictions)

    loss = -np.sum(np.log(probs[target_index]))

    dprediction = probs[target_index]

    dprediction[target_index] -= 1

    return loss, dprediction
