#!/usr/bin/env python

import random

import numpy as np

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid


def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    return x / np.sqrt(np.sum(x * x, axis=1))[:, None]
    ### END YOUR CODE


def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0, 4.0], [1, 2]]))
    ans = np.array([[0.6, 0.8], [0.4472136, 0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print("")


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    y_hat = softmax(np.dot(predicted, outputVectors.T))
    cost = -np.log(y_hat[target])
    grad = np.outer(y_hat, predicted)
    grad[target] -= predicted
    gradPred = -outputVectors[target] + np.sum(y_hat[:, None] * outputVectors, axis=0)
    ### END YOUR CODE

    return cost, gradPred, grad


def test_softmaxCostAndGradient():
    Ninner = 3
    Nwords = 5
    vc = np.random.rand(Ninner)
    target = 1
    outputVectors = np.random.rand(Ninner * Nwords).reshape((Nwords, Ninner))

    def softmaxCostAndGradient_pred(vc, outputVectors):
        cost, gradPred, grad = softmaxCostAndGradient(vc, target, outputVectors, None)
        return cost, gradPred

    def softmaxCostAndGradient_(vc, outputVectors):
        cost, gradPred, grad = softmaxCostAndGradient(vc, target, outputVectors, None)
        return cost, grad

    # gradcheck_naive(lambda outputVectors: softmaxCostAndGradient_(vc, outputVectors), outputVectors)
    gradcheck_naive(lambda vc: softmaxCostAndGradient_pred(vc, outputVectors), vc)


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    negative_samples = getNegativeSamples(target, dataset, K)
    indices.extend(negative_samples)

    ### YOUR CODE HERE
    negativeVectors = outputVectors[negative_samples]

    sig_target = sigmoid(np.dot(outputVectors[target].T, predicted))
    u_vc = np.sum(negativeVectors * predicted, axis=1)
    sig_negative = sigmoid(-u_vc)
    cost = - np.log(sig_target) - np.sum(np.log(sig_negative))

    gradPred = -(1 - sig_target) * outputVectors[target] \
               + np.sum((1. - sig_negative)[:, None] * negativeVectors, axis=0)
    grad = np.zeros(outputVectors.shape)
    grad[target] = - (1 - sig_target) * predicted
    negative_grad = np.outer((1 - sig_negative), predicted)
    for i, _grad in zip(negative_samples, negative_grad):
        grad[i] += _grad
    ### END YOUR CODE

    return cost, gradPred, grad


def test_negSampling():
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], \
               [tokens[random.randint(0, 4)] for i in range(2 * C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    Ninner = 3
    Nwords = 5
    vc = np.random.rand(Ninner)
    target = 1
    outputVectors = np.random.rand(Ninner * Nwords).reshape((Nwords, Ninner))

    def negSampling_pred(vc, outputVectors):
        cost, gradPred, grad = negSamplingCostAndGradient(vc, target, outputVectors, dataset)
        return cost, gradPred

    def negSampling_(vc, outputVectors):
        cost, gradPred, grad = negSamplingCostAndGradient(vc, target, outputVectors, dataset)
        return cost, grad

    # gradcheck_naive(lambda outputVectors: negSampling_(vc, outputVectors), outputVectors)
    gradcheck_naive(lambda vc: negSampling_pred(vc, outputVectors), vc)


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    contextIndices = [tokens[word] for word in contextWords]
    centerIndex = tokens[currentWord]

    # print("shape: inputVectors: {}, outputVectors: {}".format(inputVectors.shape, outputVectors.shape))

    for word in contextWords:
        target = tokens[word]
        c_, gradPred_, grad_ = word2vecCostAndGradient(predicted=inputVectors[centerIndex],
                                                       target=target,
                                                       outputVectors=outputVectors, dataset=dataset)
        cost += c_
        gradIn[centerIndex] += gradPred_
        gradOut += grad_
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    # raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:int(N / 2), :]
    outputVectors = wordVectors[int(N / 2):, :]
    for i in range(batchsize):
        C1 = random.randint(1, C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:int(N / 2), :] += gin / batchsize / denom
        grad[int(N / 2):, :] += gout / batchsize / denom

    # print("shapes: cost: {} grad {}".format(cost.shape, grad.shape))
    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], \
               [tokens[random.randint(0, 4)] for i in range(2 * C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])
    # print("==== Gradient check for skip-gram ====")
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
    #     skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
    #                 dummy_vectors)
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
    #     skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
    #                 dummy_vectors)
    # print("\n==== Gradient check for CBOW      ====")
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
    #     cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
    #                 dummy_vectors)
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
    #     cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
    #                 dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
                   dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset))
    print(skipgram("c", 1, ["a", "b"],
                   dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
                   negSamplingCostAndGradient))
    print(cbow("a", 2, ["a", "b", "c", "a"],
               dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset))
    print(cbow("a", 2, ["a", "b", "a", "c"],
               dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
               negSamplingCostAndGradient))


if __name__ == "__main__":
    # test_normalize_rows()
    # test_softmaxCostAndGradient()
    # test_negSampling()
    test_word2vec()
