#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    l1 = np.dot(X, W1) + b1
    z1 = sigmoid(l1)

    l2 = np.dot(z1, W2) + b2
    z2 = softmax(l2)

    def logloss(y, y_):
        return -np.sum(y * np.log(y_))

    cost = logloss(labels, z2)

    ### YOUR CODE HERE: backward propagation
    grad = labels - z2
    softmax_grad = grad * (1 - grad)
    print("z1: {}, softmax_grad: {}, W2: {}, b2: {}".format(z1.shape, softmax_grad.shape, W2.shape, b2.shape))
    gradW2 = np.dot(z1.T, softmax_grad)
    gradb2 = np.reshape(np.mean(softmax_grad, axis=0), b2.shape)
    print("gradW2 {} gradb2 {}".format(gradW2.shape, gradb2.shape))
    assert gradW2.shape == W2.shape
    assert gradb2.shape == b2.shape
    # d y / d w1 = d z2 / d w1 = d z2 / d l2 * d l2 / d z1 * d z1 / d l1 * d l1 / d w1
    # softmax_grad * W2 * sigmoid_grad * X
    dy_dl2 = np.dot(softmax_grad, W2.T)
    print("dy_dl2: {}".format(dy_dl2.shape))

    sig_grad = sigmoid_grad(dy_dl2)
    gradW1 = np.dot(X.T, sig_grad)
    gradb1 = np.reshape(np.mean(sig_grad, axis=0), b1.shape)
    print("gradW1 {} gradb1 {}".format(gradW1.shape, gradb1.shape))
    assert gradW1.shape == W1.shape
    assert gradb1.shape == b1.shape
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
                           gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])  # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0, dimensions[2] - 1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
            dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
                    forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
