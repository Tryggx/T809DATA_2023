import numpy as np
import matplotlib.pyplot as plt


def normal(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    # Part 1.1
    return 1/(np.sqrt(sigma**2*2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)

def plot_normal(sigma: float, mu:float, x_start: float, x_end: float):
    # Part 1.2
    x = np.linspace(x_start, x_end, 500)
    y = normal(x, sigma, mu)
    plt.plot(x, y)

def _plot_three_normals():
    # Part 1.2
    pass

def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
    # Part 2.1
    result = 0
    for i in range(len(sigmas)):
        result += weights[i]/(np.sqrt((sigmas[i]**2)*2*np.pi))*np.exp(-(((x-mus[i])**2)/(2*sigmas[i]**2)))
    return result

def _compare_components_and_mixture():
    # Part 2.2
    pass

def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
    # Part 3.1
    pass

def _plot_mixture_and_samples():
    # Part 3.2
    pass

if __name__ == '__main__':
    # select your function to test here and do `python3 template.py`
    # part 1.1
    '''x = np.array([-1, 0, 1])
    sigma = 1
    mu = 0
    #print(normal(x, sigma, mu))'''

    # part 1.2
    '''
    sigma = [0.5,0.25,1]
    mu = [0,1,1]
    plt.figure()
    for i in range(3):
        plot_normal(sigma[i], mu[i], -2, 2)
    plt.show()
    '''
    # part 2.1
    '''
    x = np.linspace(-5, 5, 5)
    sigmas = [0.5, 0.25, 1]
    mus = [0, 1, 1.5]
    weights = [1/3, 1/3, 1/3]
    print(normal_mixture(x, sigmas, mus, weights))
    '''
    # part 2.2
    mu = [0, -0.5, 1.5]
    sigma = [0.5, 1.5, 0.25]
    weights = [1/3, 1/3, 1/3]