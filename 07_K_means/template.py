# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.



import numpy as np
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import Union

from tools import load_iris, image_to_numpy, plot_gmm_results


def distance_matrix(
    X: np.ndarray,
    Mu: np.ndarray
) -> np.ndarray:
    '''
    Returns a matrix of euclidian distances between points in
    X and Mu.

    Input arguments:
    * X (np.ndarray): A [n x f] array of samples
    * Mu (np.ndarray): A [k x f] array of prototypes

    Returns:
    out (np.ndarray): A [n x k] array of euclidian distances
    where out[i, j] is the euclidian distance between X[i, :]
    and Mu[j, :]
    '''
    D = np.zeros((X.shape[0], Mu.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Mu.shape[0]):
            D[i, j] = np.linalg.norm(X[i, :] - Mu[j, :])
    return D


def determine_r(dist: np.ndarray) -> np.ndarray:
    '''
    Returns a matrix of binary indicators, determining
    assignment of samples to prototypes.

    Input arguments:
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    out (np.ndarray): A [n x k] array where out[i, j] is
    1 if sample i is closest to prototype j and 0 otherwise.
    '''
    A = np.zeros(dist.shape)
    for i in range(dist.shape[0]):
        A[i, np.argmin(dist[i, :])] = 1
    return A


def determine_j(R: np.ndarray, dist: np.ndarray) -> float:
    '''
    Calculates the value of the objective function given
    arrays of indicators and distances.

    Input arguments:
    * R (np.ndarray): A [n x k] array where out[i, j] is
        1 if sample i is closest to prototype j and 0
        otherwise.
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    * out (float): The value of the objective function
    '''
    total_dist = np.sum(R*dist)
    return total_dist/np.sum(R)


def update_Mu(
    Mu: np.ndarray,
    X: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    '''
    Updates the prototypes, given arrays of current
    prototypes, samples and indicators.

    Input arguments:
    Mu (np.ndarray): A [k x f] array of current prototypes.
    X (np.ndarray): A [n x f] array of samples.
    R (np.ndarray): A [n x k] array of indicators.

    Returns:
    out (np.ndarray): A [k x f] array of updated prototypes.
    '''
    Mu_new = np.zeros(Mu.shape)
    for i in range(Mu.shape[0]):
        Mu_new[i, :] = np.sum(R[:, i].reshape(-1, 1)*X, axis=0)/np.sum(R[:, i])
    return Mu_new


def k_means(
    X: np.ndarray,
    k: int,
    num_its: int
) -> Union[list, np.ndarray, np.ndarray]:
    # We first have to standardize the samples
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = (X-X_mean)/X_std
    # run the k_means algorithm on X_st, not X.

    # we pick K random samples from X as prototypes
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu = X_standard[nn[0: k], :]

    j = np.zeros(num_its)
    for i in range(num_its):
        # We calculate the distance matrix
        D = distance_matrix(X_standard, Mu)
        # We determine the indicators
        R = determine_r(D)
        J = determine_j(R, D)
        # We update the prototypes
        Mu = update_Mu(Mu, X_standard, R)
        j[i] = J

    # Then we have to "de-standardize" the prototypes
    for i in range(k):
        Mu[i, :] = Mu[i, :] * X_std + X_mean

    return Mu, R, j

def _plot_j():
    plt.figure()
    X, y, c = load_iris()
    _,_,j = k_means(X, 4, 10)
    plt.plot(j)
    plt.xlabel('Iteration')
    plt.ylabel('J')
    #plt.show()


def _plot_multi_j():
    plt.figure()
    k = [2,3,5,10]
    X, y, c = load_iris()
    for i in k:
        _,_,j = k_means(X, i, 10)
        plt.plot(j, label = f'k = {i}')
    plt.xlabel('Iteration')
    plt.ylabel('J')
    plt.legend()
    #plt.show()


def k_means_predict(
    X: np.ndarray,
    t: np.ndarray,
    classes: list,
    num_its: int
) -> np.ndarray:
    '''
    Determine the accuracy and confusion matrix
    of predictions made by k_means on a dataset
    [X, t] where we assume the most common cluster
    for a class label corresponds to that label.

    Input arguments:
    * X (np.ndarray): A [n x f] array of input features
    * t (np.ndarray): A [n] array of target labels
    * classes (list): A list of possible target labels
    * num_its (int): Number of k_means iterations

    Returns:
    * the predictions (list)
    '''
    _, R, _ = k_means(X, len(classes), num_its)
    pred = np.zeros(t.shape)
    for i in range(len(classes)):
        cluster = np.argmax(np.sum(R[t == classes[i], :], axis = 0))
        pred[R[:, cluster] == 1] = classes[i]

        
    return pred


def _iris_kmeans_accuracy():
    X, y, c = load_iris()
    pred = k_means_predict(X, y, c, 5)
    print(accuracy_score(y, pred))
    print(confusion_matrix(y, pred))


def _my_kmeans_on_image():
    image, (w, h) = image_to_numpy()
    print(image.shape)
    output = k_means(image, 7, 5)
    # plt.imshow(output[0].reshape(w, h, 3))
    # plt.show()


def plot_image_clusters(n_clusters: int):
    '''
    Plot the clusters found using sklearn k-means.
    '''
    image, (w, h) = image_to_numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image)
    plt.subplot(121)
    plt.imshow(image.reshape(w, h, 3))
    plt.subplot(122)
    # uncomment the following line to run
    plt.imshow(kmeans.labels_.reshape(w, h), cmap="plasma")
    plt.show()


if '__main__' == __name__:
#1.1
    a = np.array([
        [1, 0, 0],
        [4, 4, 4],
        [2, 2, 2]])
    b = np.array([
        [0, 0, 0],
        [4, 4, 4]])
    #print(distance_matrix(a, b))

#1.2
    dist = np.array([
        [  1,   2,   3],
        [0.3, 0.1, 0.2],
        [  7,  18,   2],
        [  2, 0.5,   7]])
    #print(determine_r(dist))

#1.3
    dist = np.array([
            [  1,   2,   3],
            [0.3, 0.1, 0.2],
            [  7,  18,   2],
            [  2, 0.5,   7]])
    R = determine_r(dist)
    #print(determine_j(R, dist))

#1.4
    X = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]])
    Mu = np.array([
        [0.0, 0.5, 0.1],
        [0.8, 0.2, 0.3]])
    R = np.array([
        [1, 0],
        [0, 1],
        [1, 0]])
    #print(update_Mu(Mu, X, R))

#1.5

    X, y, c = load_iris()
    #print(k_means(X, 4, 10))

#1.6
    _plot_j()

#1.7
    _plot_multi_j()

    #plt.show()


#1.9
    X, y, c = load_iris()
    print(k_means_predict(X, y, c, 5))


#1.10
    _iris_kmeans_accuracy()

#2.1 
    #_my_kmeans_on_image()
    # plot_image_clusters(2)
    # plot_image_clusters(5)
    # plot_image_clusters(10)
    # plot_image_clusters(20)
