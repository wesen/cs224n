#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def crossentropy_softmax_grad(Y, softmax_theta):
    """
    Compute the gradient of the cross entropy of the softmax according to the input vector theta.
    softmax_theta is the result of the softmax applied to theta.

    Y is a one hot encoded vector.
    """
    return softmax_theta - Y


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
    z1 = np.dot(X, W1) + b1
    # print("W1 (python): {}".format(W1))
    # print("b1 (python): {}".format(b1))
    # print("z1 (python): {}".format(z1))
    l1 = sigmoid(z1)

    z2 = np.dot(l1, W2) + b2
    l2 = softmax(z2)

    print("Cost: {}".format(-np.sum(labels * np.log(l2), axis=1)))
    cost = -np.sum(labels * np.log(l2))

    ### YOUR CODE HERE: backward propagation
    softmax_grad = crossentropy_softmax_grad(labels, l2)
    print("softmax_grad: {}".format(softmax_grad))
    # print("z1: {}, softmax_grad: {}, W2: {}, b2: {}".format(z1.shape, softmax_grad.shape, W2.shape, b2.shape))
    gradW2 = np.dot(z1.T, softmax_grad)
    gradb2 = np.reshape(np.mean(softmax_grad, axis=0), b2.shape)
    print("gradW2 (python): {}".format(gradW2))

    # print("gradW2 {} gradb2 {}".format(gradW2.shape, gradb2.shape))
    assert gradW2.shape == W2.shape
    assert gradb2.shape == b2.shape
    # d y / d w1 = d z2 / d w1 = d z2 / d l2 * d l2 / d z1 * d z1 / d l1 * d l1 / d w1
    # softmax_grad * W2 * sigmoid_grad * X
    dy_dl2 = np.dot(softmax_grad, W2.T)
    # print("dy_dl2: {}".format(dy_dl2.shape))

    sig_grad = sigmoid_grad(dy_dl2)
    gradW1 = np.dot(X.T, sig_grad)
    gradb1 = np.reshape(np.mean(sig_grad, axis=0), b1.shape)
    # print("gradW1 {} gradb1 {}".format(gradW1.shape, gradb1.shape))
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
    import h5py
    import glob
    sample_files = glob.glob("C:\\tmp\\sample*.h5")
    for file in sample_files:
        with h5py.File(file) as f:
            data = f["Input"][...]
            labels = f["Target"][...]
            N = data.shape[1]
            dimensions = [2, 3, 3]
            l1Weights = f["L1Weights"][...]
            l1Biases = f["L1Biases"][...]
            l2Weights = f["L2Weights"][...]
            l2Biases = f["L2Biases"][...]
            loss = f["Loss"][...]
            z1 = f["z1"][...]
            l1 = f["l1"][...]
            params = np.hstack([l1Weights.T.flatten(), l1Biases.flatten(), l2Weights.T.flatten(), l2Biases.flatten()])
            print("gradW2 (wl): {}".format(f["GradientL2Weights"][...]))
            print("loss: {}".format(loss))
            cost, grad = forward_backward_prop(data, labels, params, dimensions)
            print("cost: {}, should: {}".format(cost, np.sum(loss)))



if __name__ == "__main__":
    # sanity_check()
    your_sanity_checks()
