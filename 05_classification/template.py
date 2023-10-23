# Author: Friðrik Tryggvi Róbertsson
# Date: 2023-09-29
# Project: 05_classification
# Acknowledgements: 
#


from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    mean_list = []
    for i in range(features.shape[0]):
        if targets[i] == selected_class:
            mean_list.append(features[i, :])
    return np.mean(mean_list, axis=0)


def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    covar_list = []
    for i in range(features.shape[0]):
        if targets[i] == selected_class:
            covar_list.append(features[i, :])
    return np.cov(np.array(covar_list).T)


def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    return multivariate_normal.pdf(feature, class_mean, class_covar)


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))
        covs.append(covar_of_class(train_features, train_targets, class_label))
    likelihoods = []
    for i in range(test_features.shape[0]):
        likelihoods.append([])
        for j in range(len(classes)):
            likelihoods[i].append(likelihood_of_class(test_features[i, :], means[j], covs[j]))
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    return np.argmax(likelihoods, axis=1)


def maximum_aposteriori(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum a posteriori for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
    priors = []
    for class_label in classes:
        priors.append(train_targets[train_targets == class_label].shape[0] / train_targets.shape[0])
    # Calculate the posterior probability of each class
    for i in range(len(likelihoods)):
        for j in range(len(likelihoods[i])):
            likelihoods[i][j] = likelihoods[i][j] * priors[j]
    return np.array(likelihoods)

def confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the confusion matrix for a set of predictions
    and targets.
    '''
    matrix = np.zeros((len(classes), len(classes)))
    for i in range(len(predictions)):
        matrix[predictions[i], targets[i]] += 1
    return matrix
            


if __name__ == '__main__':
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = split_train_test(
        features, targets, 0.6
    )
    print(mean_of_class(train_features, train_targets, 0))
    class_mean = mean_of_class(train_features, train_targets, 0)

    print(covar_of_class(train_features, train_targets, 0))
    class_covar = covar_of_class(train_features, train_targets, 0)

    print(likelihood_of_class(train_features[0, :], class_mean, class_covar))

    print(maximum_likelihood(train_features, train_targets, test_features, classes))

    #Maximum likelihood
    likelihoods = maximum_likelihood(
        train_features, train_targets, test_features, classes
    )
    predictions = predict(likelihoods)
    print(predictions)
    print(test_targets)
    print('likelihood: ',np.sum(predictions == test_targets) / len(test_targets))
    print(confusion_matrix(predictions, test_targets, classes))

    # Maximum a posteriori
    likelihoods = maximum_aposteriori(
        train_features, train_targets, test_features, classes
    )
    predictions = predict(likelihoods)
    print(predictions)
    print(test_targets)

    #accuracy of maximum likelihood and maximum a posteriori
    print('a-posteriori: ',np.sum(predictions == test_targets) / len(test_targets))
    print(confusion_matrix(predictions, test_targets, classes))
