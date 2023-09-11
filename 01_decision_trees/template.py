# Author: 
# Date:
# Project: 
# Acknowledgements: 
#


from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tools import load_iris, split_train_test



def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    count = 0
    list = []
    try: 
        targets = np.ndarray.tolist(targets)
    except:
        pass
    for c in classes:
        count = targets.count(c)
        list.append(count/len(targets))
    
    return list


def split_data(
    features: np.ndarray,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    features_1 = []
    features_2 = []
    targets_1 = []
    targets_2 = []
    for i in range(len(features)):
        if features[i][split_feature_index] < theta:
            features_1.append(features[i][split_feature_index])
            targets_1.append(targets[i])
        else:
            features_2.append(features[i][split_feature_index])
            targets_2.append(targets[i])

    return (features_1, targets_1), (features_2, targets_2)


def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    priors = prior(targets, classes)
    sum_squared_priors = sum([p ** 2 for p in priors])
    
    # Calculate the Gini impurity using the formula
    impurity = 0.5 * (1 - sum_squared_priors)
    
    return impurity



def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
   
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2, classes)
    t1 = np.array(t1)
    t2 = np.array(t2)
    n = t1.shape[0] + t2.shape[0]
    
    # Calculate the weighted impurity using the formula
    impurity = (t1.shape[0]/n) * g1 + (t2.shape[0]/n) * g2
    return impurity


def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''
    # Split the data
    (f_1, t_1), (f_2, t_2) = split_data(features, targets, split_feature_index, theta)
    # Calculate the weighted impurity using the formula
    impurity = weighted_impurity(t_1, t_2, classes)
    return impurity


def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None
    # iterate feature dimensions
    for i in range(features.shape[1]):
        # create the thresholds
        thetas = np.linspace(features[:, i].min(), features[:, i].max(), num_tries+2)[1:-1]
        # iterate thresholds
        for theta in thetas:
            # calculate the gini impurity
            gini = total_gini_impurity(features, targets, classes, i, theta)
            # update the best values
            if gini < best_gini:
                best_gini = gini
                best_dim = i
                best_theta = theta
    return best_gini, best_dim, best_theta


class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        self.tree.fit(self.train_features, self.train_targets)

    def accuracy(self):
        return self.tree.score(self.test_features, self.test_targets)

    def plot(self):
        plot_tree(self.tree, filled=True) #feature_names=['sepal length', 'sepal width', 'petal length', 'petal width'], class_names=['setosa', 'versicolor', 'virginica'])
        #plt.show()

    def plot_progress(self):
        # Independent section
        # Remove this method if you don't go for independent section.
        ...

    def guess(self):
        return self.tree.predict(self.test_features)

    def confusion_matrix(self):
        #create a confusion matrix
        matrix = np.array([[0,0,0],[0,0,0],[0,0,0]])
        #iterate over the test targets and the guesses
        for i in range(len(self.test_targets)):
            matrix[self.test_targets[i]][self.guess()[i]] += 1
        return matrix

if __name__ == '__main__':
    # print(np.array([0,0,1]))
    # print(prior([0,0,1],[0,1]))
    # print(prior([0,2,3,3],[0,1,2,3]))
    # features, targets, classes = load_iris()
    # (f_1, t_1), (f_2, t_2) = split_data(features, targets, 2, 4.65)
    # print(len(f_1), len(f_2))

    # print(gini_impurity(t_1, classes))
    # print(gini_impurity(t_2, classes))
    # print(weighted_impurity(t_1, t_2, classes)) 
    # print(total_gini_impurity(features, targets, classes, 2, 4.65))
    # print(brute_best_split(features, targets, classes, 30))

    # dt = IrisTreeTrainer(features, targets, classes=classes)
    # dt.train()
    # print(f'The accuracy is: {dt.accuracy()}')
    # dt.plot()
    # print(f'I guessed: {dt.guess()}')
    # print(f'The true targets are: {dt.test_targets}')
    # print(dt.confusion_matrix())
    pass