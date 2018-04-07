#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def crossentropy_softmax_grad(Y, s):
    """
    Compute the gradient of the cross entropy of the softmax according to the input vector theta.
    softmax_theta is the result of the softmax applied to theta.

    Y is a one hot encoded vector.
    """
    return s - Y


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

    cost = -np.sum(labels * np.log(l2))
    print("Cost: {} ({})".format(-np.sum(labels * np.log(l2), axis=1), cost))

    ### YOUR CODE HERE: backward propagation
    softmax_grad = crossentropy_softmax_grad(labels, l2)
    # print("softmax_grad: {}".format(softmax_grad))
    print("shapes: z1: {}, softmax_grad: {}, W2: {}, b2: {}".format(z1.shape, softmax_grad.shape, W2.shape, b2.shape))
    gradW2 = np.dot(l1.T, softmax_grad)
    gradb2 = np.reshape(np.mean(softmax_grad, axis=0), b2.shape)
    # print("gradW2 (python): {}".format(gradW2))

    print("shapes: gradW2 {} gradb2 {}".format(gradW2.shape, gradb2.shape))
    assert gradW2.shape == W2.shape
    assert gradb2.shape == b2.shape
    # d y / d w1 = d z2 / d w1 = d z2 / d l2 * d l2 / d z1 * d z1 / d l1 * d l1 / d w1
    # softmax_grad * W2 * sigmoid_grad * X
    # print("dy_dl2: {}".format(dy_dl2.shape))

    sig_grad = sigmoid_grad(l1)
    dy_dz1 = sig_grad * np.dot(softmax_grad, W2.T)
    gradW1 = np.dot(X.T, dy_dz1)
    gradb1 = np.reshape(np.mean(dy_dz1, axis=0), b1.shape)
    # print("gradW1 {} gradb1 {}".format(gradW1.shape, gradb1.shape))
    assert gradW1.shape == W1.shape
    assert gradb1.shape == b1.shape
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
                           gradW2.flatten(), gradb2.flatten()))

    return cost, grad

def softmax_ce_loss(X, Y):
    sX = softmax(X)
    cost = -np.sum(Y * np.log(sX))

    grad = crossentropy_softmax_grad(Y, sX)
    print("grad: {}".format(grad))

    return cost, grad

def softmax_ce_loss_w(X, params, Y, dimensions):
    Dx, H, Dy = dimensions

    W = np.reshape(params[:H * Dy], (H, Dy))

    sX = softmax(np.dot(X, W))
    print("shapes: X {}, W {}, Y {}, sX: {}".format(X.shape, W.shape, Y.shape, sX.shape))
    cost = -np.sum(Y * np.log(sX))

    grad = np.dot(X.T, crossentropy_softmax_grad(Y, sX))
    print("grad: {}, grad_f {}, params {}".format(grad.shape, grad.flatten().shape, params.shape))

    return cost, grad.flatten()

def softmax_ce_loss_w_sigmoid(X, params, Y, dimensions):
    Dx, H, Dy = dimensions

    l1 = sigmoid(X)

    W2 = np.reshape(params[:H * Dy], (H, Dy))

    sX = softmax(np.dot(l1, W2))
    print("shapes: X {}, W {}, Y {}, sX: {}".format(X.shape, W2.shape, Y.shape, sX.shape))
    cost = -np.sum(Y * np.log(sX))

    softmax_grad = crossentropy_softmax_grad(Y, sX)
    grad = sigmoid_grad(l1) * np.dot(softmax_grad, W2.T)
    print("grad: {}, grad_f {}, params {}".format(grad.shape, grad.flatten().shape, params.shape))

    return cost, grad

def softmax_ce_loss_w_w2_sigmoid(X, params, Y, dimensions):
    Dx, H, Dy = dimensions
    print(Dx,H,Dy,params.shape)

    X = sigmoid(X)

    W2 = np.reshape(params[Dx * H:H * Dy + Dx * H], (H, Dy))
    W = np.reshape(params[:Dx * H], (Dx, H))

    z1 = np.dot(X, W)
    l1 = sigmoid(z1)
    z2 = np.dot(l1, W2)
    sX = softmax(z2)
    cost = -np.sum(Y * np.log(sX))
    print("shapes: X {}, W {}, Y {}, sX: {}, cost: {}".format(X.shape, W.shape, Y.shape, sX.shape, cost))

    softmax_grad = crossentropy_softmax_grad(Y, sX)
    dydz1 = sigmoid_grad(l1) * np.dot(softmax_grad, W2.T)
    grad = np.dot(X.T, dydz1)
    print("grad: {}, grad_f {}, params {}".format(grad.shape, grad.flatten().shape, params.shape))

    return cost, grad.flatten()

def ce_loss(X, Y):
    cost = -np.sum(Y * np.log(X))
    grad = -Y / X
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

    N = 2
    dimensions = [3, 5, 2]
    data = 10. * np.random.randn(N, dimensions[0])  # each row will be a datum
    data2 = 10. * np.random.randn(N, dimensions[1])  # each row will be a datum
    data3 = np.random.randn(N, dimensions[2]) /2. + .5  # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0, dimensions[2] - 1)] = 1

    W2 = np.random.randn(dimensions[1] * dimensions[2])
    W = np.random.randn(dimensions[0] * dimensions[1])

    # print("\nsoftmax_ce_loss\n")
    # gradcheck_naive(lambda x: softmax_ce_loss(x, labels), data3)
    # print("\nsoftmax_ce\n")
    # gradcheck_naive(lambda x: ce_loss(x, labels), data3)
    # print("\nsoftmax_ce_loss_w\n")
    # gradcheck_naive(lambda params:
    #                 softmax_ce_loss_w(data2, params, labels, dimensions), W2)
    # print("\nsoftmax_ce_loss_w_sigmoid\n")
    gradcheck_naive(lambda x:
                    softmax_ce_loss_w_sigmoid(x, W2, labels, dimensions), data2)
    print("\nsoftmax_ce_loss_w_w2_sigmoid\n")
    gradcheck_naive(lambda W:
                    softmax_ce_loss_w_w2_sigmoid(data, np.concatenate((W, W2)), labels, dimensions), W)
    return

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
    sanity_check()
    # your_sanity_checks()
