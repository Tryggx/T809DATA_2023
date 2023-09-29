# Author: Friðrik Tryggvi Róbertsson
# Date: 11/09/2023
# Project: 3 sequential estimation
# Acknowledgements: 
#


from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    
    X = np.zeros((n, k))
    for i in range(n):
        X[i, :] = np.random.multivariate_normal(mean, var**2*np.eye(k))
    return X


def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    return mu + (1.0/n)*(x-mu)


def _plot_sequence_estimate():
    data = X # Set this as the data
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[-1], data[i, :], i+1))
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    plt.legend(loc='upper center')
    plt.show()


def _square_error(y, y_hat):
    return np.sum((y-y_hat)**2)


def _plot_mean_square_error():
    data = X # Set this as the data
    estimates = [np.array([0, 0, 0])]
    errors = []
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[-1], data[i, :], i+1))
        errors.append(_square_error(mean, estimates[-1]))
    plt.plot(errors)
    plt.show()


# Naive solution to the independent question.

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: float
) -> np.ndarray:
    # remove this if you don't go for the independent section
    pass


def _plot_changing_sequence_estimate():
    # remove this if you don't go for the independent section
    pass


if __name__ == '__main__':
    pass
    # #remove this if you don't go for the independent section
    # #part 1
    # np.random.seed(1234)
    # print(gen_data(2, 3, np.array([0, 1, -1]), 1.3))
    # np.random.seed(1234)
    # print(gen_data(5, 1, np.array([0.5]), 0.5))

    # # part 2
    # np.random.seed(1234)
    # X = gen_data(300, 3, np.array([0, 1, -1]), np.sqrt(3))
    # #save x as txt file

    # np.random.seed(1234)
    # scatter_3d_data(X)
    # bar_per_axis(X)

    # #part 4
    # mean = np.mean(X, 0)
    # np.random.seed(1234)
    # new_x = gen_data(1, 3, np.array([0, 0, 0]), 1)
    # np.random.seed(1234)
    # print(update_sequence_mean(mean, new_x, X.shape[0]))

    # #part 5
    # np.random.seed(1234)
    # X = gen_data(100, 3, np.array([0, 0, 0]), 4)
    # _plot_sequence_estimate()

    # #part 6
    # _plot_mean_square_error()
