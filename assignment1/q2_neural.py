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
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])  # 10, 5, 10

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))  # 10, 5
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))  # 1, 5
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))  # 5, 10
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))  # 1, 10

    # Note: compute cost based on `sum` not `mean`.
    ### YOUR CODE HERE: forward propagation
    z2 = X.dot(W1) + b1  # (20, 5) - 20 is the number if training examples
    a2 = sigmoid(z2)  # (20, 5)
    z3 = a2.dot(W2) + b2  # (20, 10)
    a3 = softmax(z3)  # (20, 10)
    cost = -np.sum(labels * np.log(a3))  # cross entropy cost
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    delta3 = a3 - labels  # 20, 10 - the derivative of cross entropy
    gradb2 = np.sum(delta3, 0, keepdims=True)  # summing over training examples (1,  10), derivative over b is 1
    gradW2 = np.dot(a2.T, delta3)  # works similar to the derivative with respect to input x or hidden layer h

    delta2 = sigmoid_grad(a2) * np.dot(delta3, W2.T)  # see assign1, 2(c)
    gradb1 = np.sum(delta2, 0, keepdims=True)
    gradW1 = np.dot(X.T, delta2)
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
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

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
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    N = 50
    dimensions = [10, 20, 5]
    data = np.random.randn(N, dimensions[0])  # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0, dimensions[2] - 1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
            dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
                    forward_backward_prop(data, labels, params, dimensions), params)
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
