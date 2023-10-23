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
from sklearn.decomposition import PCA

from tools import load_cancer


def standardize(X: np.ndarray) -> np.ndarray:
    '''
    Standardize an array of shape [N x 1]

    Input arguments:
    * X (np.ndarray): An array of shape [N x 1]

    Returns:
    (np.ndarray): A standardized version of X, also
    of shape [N x 1]
    '''
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def scatter_standardized_dims(
    X: np.ndarray,
    i: int,
    j: int,
):
    '''
    Plots a scatter plot of N points where the n-th point
    has the coordinate (X_ni, X_nj)

    Input arguments:
    * X (np.ndarray): A [N x f] array
    * i (int): The first index
    * j (int): The second index
    '''
    X = standardize(X)
    plt.scatter(X[:, i], X[:, j])


def _scatter_cancer():
    X, y = load_cancer()
    for i in range(0 , 30):
                   
        plt.subplot(5,6, i+1)
        plt.scatter(X[:, 0], X[:, i])  
  
    plt.show()


def _plot_pca_components():
    # standardize X
    # fit PCA and set n_properties, ie M, to D.
    #use pca.fit_transform(X) to get the principal components
    #plot the each component on a single plot within a subplot
    X, y = load_cancer()
    x = standardize(X)
    pca = PCA(n_components=x.shape[1])
    pca.fit_transform(x)
    components = pca.components_
    for i in range(0, pca.n_components_):
        plt.subplot(5,6, i+1)
        plt.plot(components[i,:])
        plt.title(f'PCA {i+1}')
    plt.show()


def _plot_eigen_values():
    X, y = load_cancer()
    x = standardize(X)
    pca = PCA()
    pca.n_components_ = x.shape[1]
    pca.fit_transform(x)
    print(pca.explained_variance_)
    eigen_index = np.arange(0, pca.explained_variance_.shape[0])
    plt.plot(eigen_index, pca.explained_variance_)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue')
    plt.grid()


def _plot_log_eigen_values():
    X, y = load_cancer()
    x = standardize(X)
    pca = PCA()
    pca.n_components_ = x.shape[1]
    pca.fit_transform(x)
    eigen_index = np.arange(0, pca.explained_variance_.shape[0])
    plt.plot(eigen_index, np.log(pca.explained_variance_))
    plt.xlabel('Eigenvalue index')
    plt.ylabel('$\log_{10}$ Eigenvalue')
    plt.grid()
    plt.show()


def _plot_cum_variance():
    X, y = load_cancer()
    x = standardize(X)
    pca = PCA()
    pca.n_components_ = x.shape[1]
    pca.fit_transform(x)
    eigen_index = np.arange(0, pca.explained_variance_.shape[0])
    plt.plot(eigen_index, np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Percentage variance')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    
    # print(standardize([[0,0], [0,0], [1,1], [1,1]]))
    # X = np.array([
    # [1, 2, 3, 4],
    # [0, 0, 0, 0],
    # [4, 5, 5, 4],
    # [2, 2, 2, 2],
    # [8, 6, 4, 2]])
    # scatter_standardized_dims(X, 0, 2)
    #_scatter_cancer()
    # _plot_pca_components()
    #_plot_eigen_values()
    #_plot_log_eigen_values()
    _plot_cum_variance()
    plt.show()
