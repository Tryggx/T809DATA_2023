# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, RandomizedSearchCV)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, precision_score)

from tools import get_titanic, build_kaggle_submission


def get_better_titanic():
    '''
    Loads the cleaned titanic dataset but change
    how we handle the age column.
    '''
    # Load in the raw data
    if os.path.exists('./data/train.csv'):
        train = pd.read_csv('./data/train.csv')
        test = pd.read_csv('./data/test.csv')
    else:
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
     # Concatenate the train and test set into a single dataframe
    # we drop the `Survived` column from the train set
    X_full = pd.concat([train.drop('Survived', axis=1), test], axis=0)

    # The cabin category consist of a letter and a number.
    # We can divide the cabin category by extracting the first
    # letter and use that to create a new category. So before we
    # drop the `Cabin` column we extract these values
    X_full['Cabin_mapped'] = X_full['Cabin'].astype(str).str[0]
    # Then we transform the letters into numbers
    cabin_dict = {k: i for i, k in enumerate(X_full.Cabin_mapped.unique())}
    X_full.loc[:, 'Cabin_mapped'] =\
        X_full.loc[:, 'Cabin_mapped'].map(cabin_dict)

    # We drop multiple columns that contain a lot of NaN values
    # in this assignment
    # Maybe we should
    X_full.drop(
        ['PassengerId', 'Cabin', 'Name', 'Ticket'],
        inplace=True, axis=1)

    X_full['Age'].fillna(X_full['Age'].mean(), inplace=True)
    # Instead of dropping the fare column we replace NaN values
    # with the 3rd class passenger fare mean.
    fare_mean = X_full[X_full.Pclass == 3].Fare.mean()
    X_full['Fare'].fillna(fare_mean, inplace=True)
    # Instead of dropping the Embarked column we replace NaN values
    # with `S` denoting Southampton, the most common embarking
    # location
    X_full['Embarked'].fillna('S', inplace=True)

    # We then use the get_dummies function to transform text
    # and non-numerical values into binary categories.
    X_dummies = pd.get_dummies(
        X_full,
        columns=['Sex', 'Cabin_mapped', 'Embarked'],
        drop_first=True)

    # We now have the cleaned data we can use in the assignment
    X = X_dummies[:len(train)]
    submission_X = X_dummies[len(train):]
    y = train.Survived
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=5, stratify=y)

    return (X_train, y_train), (X_test, y_test), submission_X

def rfc_train_test(X_full, t_train, X_test, t_test):
    '''
    Train a random forest classifier on (X_full, t_train)
    and evaluate it on (X_test, t_test)
    '''
    # Instantiate the classifier
    rfc = RandomForestClassifier(
    )
    # Fit the classifier to the training data
    rfc.fit(X_full, t_train)
    # Predict on the test data
    pred = rfc.predict(X_test)
    # Print the accuracy score
    acc = accuracy_score(t_test, pred)
    precision = precision_score(t_test, pred)
    recall = recall_score(t_test, pred)
    return acc, precision, recall


def gb_train_test(X_full, t_train, X_test, t_test):
    '''
    Train a Gradient boosting classifier on (X_full, t_train)
    and evaluate it on (X_test, t_test)
    '''
    gb = GradientBoostingClassifier(
    )
    # Fit the classifier to the training data
    gb.fit(X_full, t_train)
    # Predict on the test data
    pred = gb.predict(X_test)
    acc = accuracy_score(t_test, pred)
    precision = precision_score(t_test, pred)
    recall = recall_score(t_test, pred)
    return acc, precision, recall


def param_search(X, y):
    '''
    Perform randomized parameter search on the
    gradient boosting classifier on the dataset (X, y)
    '''
    # Create the parameter grid
    gb_param_grid = {
        'n_estimators': [10, 20, 30, 40 ,50, 60, 70, 80, 90, 100],
        'max_depth': [1,2,3,4,5,10,20,30,40,50],
        'learning_rate': [0.01,0.02,0.03,0.04, 0.1,0.2,0.3,0.4,0.5,0.6]}
    # Instantiate the regressor
    gb = GradientBoostingClassifier()
    # Perform random search
    gb_random = RandomizedSearchCV(
        param_distributions=gb_param_grid,
        estimator=gb,
        scoring="accuracy",
        verbose=0,
        n_iter=50,
        cv=4)
    # Fit randomized_mse to the data
    gb_random.fit(X, y)
    # Print the best parameters and lowest RMSE
    return gb_random.best_params_


def gb_optimized_train_test(X_full, t_train, X_test, t_test):
    '''
    Train a gradient boosting classifier on (X_full, t_train)
    and evaluate it on (X_test, t_test) with
    your own optimized parameters
    '''
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.04
        
    )
    # Fit the classifier to the training data
    gb.fit(X_full, t_train)
    # Predict on the test data
    pred = gb.predict(X_test)
    acc = accuracy_score(t_test, pred)
    precision = precision_score(t_test, pred)
    recall = recall_score(t_test, pred)
    return acc, precision, recall


def _create_submission():
    '''Create your kaggle submission
    '''
    pass
    prediction = None # !!! Your prediction here !!!
    build_kaggle_submission(prediction)



if __name__ == "__main__":
    #1.1
    #print(get_better_titanic())
    
    #2.1
    (X_train, y_train), (X_test, y_test),_ = get_better_titanic()
    print(rfc_train_test(X_train, y_train, X_test, y_test))

    #2.3
    print('Default ')
    print(gb_train_test(X_train, y_train, X_test, y_test))

    #2.5
    print(param_search(X_train, y_train))

    #2.6 
    print(gb_optimized_train_test(X_train, y_train, X_test, y_test))
               