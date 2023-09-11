# Author: Friðrik Tryggvi Róbertsson
# Date: 31/08/2023
# Project: 02 K Nearest Neighbours
# Acknowledgements: 
#

import numpy as np
import matplotlib.pyplot as plt
import help

from tools import load_iris, split_train_test, plot_points


def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    distance = 0
    for i in range(x.shape[0]):
        distance += (x[i] - y[i])**2
    return np.sqrt(distance)


def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.ndarray(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x, points[i])
    return distances
    


def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    distances = euclidian_distances(x, points)
    return np.argsort(distances)[:k]



def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    # votes = np.array([0,0,0])
    # for i in range(targets.shape[0]):
    #     votes[targets[i]] += 1
    # return np.argmax(votes)
    classMap = {}
    for c in classes:
        classMap[c] = 0

    for t in targets:
        if t in classMap:
            classMap[t] += 1
    #sort the dictionary by value
    sortedClassMap = sorted(classMap.items(), key=lambda x: x[1], reverse=True)
    return sortedClassMap[0][0]

def knn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    k_nearest_points = k_nearest(x, points, k)
    return vote(point_targets[k_nearest_points], classes)


def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Apply knn to all points
    '''
    predictions = np.ndarray(point_targets.shape[0])
    for i in range(point_targets.shape[0]):
        points2 = help.remove_one(points, i)
        point_targets2 = help.remove_one(point_targets, i)
        predictions[i] = knn(points[i], points2, point_targets2, classes, k)
    return predictions
        


def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    '''
    Calculate the accuracy of knn
    '''
    predictions = knn_predict(points, point_targets, classes, k)
    return np.sum(predictions == point_targets) / point_targets.shape[0]


def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Calculate the confusion matrix of knn
    '''
    predictions = knn_predict(points, point_targets, classes, k)
    confusion_matrix = np.zeros((len(classes), len(classes)))
    for i in range(len(point_targets)):
        confusion_matrix[int(predictions[i]),point_targets[i]] += 1
    return confusion_matrix

def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    '''
    Find the best k for knn
    '''
    best_k = 0
    best_accuracy = 0
    for k in range(1, len(point_targets)):
        accuracy = knn_accuracy(points, point_targets, classes, k)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    return best_k


def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    colors = ['yellow', 'purple', 'blue']

    for i in range(len(point_targets)):
        predictions = knn_predict(points, point_targets, classes, k)
        if predictions[i] == point_targets[i]:
            #plot edges green
            plt.scatter(points[i][0], points[i][1], c=colors[point_targets[i]], edgecolors='green', linewidths=2)
        else:
             plt.scatter(points[i][0], points[i][1], c=colors[point_targets[i]], edgecolors='red', linewidths=2)
            
    plt.show()


def weighted_vote(
    targets: np.ndarray,
    distances: np.ndarray,
    classes: list
) -> int:
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    # Remove if you don't go for independent section
    ...


def wknn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    # Remove if you don't go for independent section
    ...


def wknn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    # Remove if you don't go for independent section
    ...


def compare_knns(
    points: np.ndarray,
    targets: np.ndarray,
    classes: list
):
    # Remove if you don't go for independent section
    ...

if __name__ == '__main__':
    # Remove if you don't go for independent section
    d, t, classes = load_iris()
    x, points = d[0,:], d[1:, :]
    x_target, point_targets = t[0], t[1:]
    print(euclidian_distance(x, points[0]))
    print(euclidian_distance(x, points[50]))

    print(euclidian_distances(x, points))

    print(k_nearest(x, points, 1))
    print(k_nearest(x, points, 3))

    print(vote(np.array([0,0,1,2]), np.array([0,1,2])))
    print(vote(np.array([1,1,1,1]), np.array([0,1])))

    print(knn(x, points, point_targets, classes, 1) )
    print(knn(x, points, point_targets, classes, 5))
    print(knn(x, points, point_targets, classes, 150))

    d, t, classes = load_iris()

    (d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)

    print(knn_predict(d_test, t_test, classes, 10))
    print(knn_predict(d_test, t_test, classes, 5))

    print(knn_accuracy(d_test, t_test, classes, 10))
    print(knn_accuracy(d_test, t_test, classes, 5))

    print(knn_confusion_matrix(d_test, t_test, classes, 10))
    print(knn_confusion_matrix(d_test, t_test, classes, 20))

    print(best_k(d_train, t_train, classes))

    knn_plot_points(d, t, classes, 3)