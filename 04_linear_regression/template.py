# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.

import numpy as np
import matplotlib.pyplot as plt

from tools import load_regression_iris
from scipy.stats import multivariate_normal


def mvn_basis(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float
) -> np.ndarray:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * sigma: All normal distributions are isotropic with sigma*I covariance
    matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    fi = np.zeros((features.shape[0], mu.shape[0]))
    for i in range(features.shape[0]):
        for j in range(mu.shape[0]):
            fi[i, j] = multivariate_normal.pdf(features[i, :], mu[j, :], sigma)
    return fi


def _plot_mvn():
    '''
    Plot the output of each basis function for each data vector
    '''
    for i in range(fi.shape[1]):
        plt.plot(fi[:, i])
    plt.show()


def max_likelihood_linreg(
    fi: np.ndarray,
    targets: np.ndarray,
    lamda: float
) -> np.ndarray:
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''
    # (fi.T * fi + lamda * I)^-1 * fi.T * t
    # covariance matrix
    covar = np.dot(fi.T, fi)
    # inverse of covariance matrix
    inverse = np.linalg.inv(covar + lamda * np.eye(fi.shape[1]))
    # weights 
    wt = np.dot(inverse, fi.T)
    # return dot product of weights and targets
    return np.dot(wt, targets)


def linear_model(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    w: np.ndarray
) -> np.ndarray:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * sigma: All normal distributions are isotropic with s*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''
    fi = mvn_basis(features, mu, sigma)
    return np.dot(fi, w)

if __name__ == '__main__':
    X, t = load_regression_iris()
    N, D = X.shape

    M, sigma = 10, 10
    mu = np.zeros((M, D))
    for i in range(D):
        mmin = np.min(X[i, :])
        mmax = np.max(X[i, :])
        mu[:, i] = np.linspace(mmin, mmax, M)
    fi = mvn_basis(X, mu, sigma)
    print(fi)
    _plot_mvn()
    lamda = 0.001
    w = max_likelihood_linreg(fi, t, lamda)
    print(w)
    wml = max_likelihood_linreg(fi, t, lamda)
    prediction = linear_model(X, mu, sigma, wml)
    print(prediction)
    mse = np.mean((prediction - t)**2)
    print(mse)

    plt.plot(t, 'r', label='targets')
    plt.plot(prediction, 'b', label='prediction')
    plt.legend()
    plt.show()
