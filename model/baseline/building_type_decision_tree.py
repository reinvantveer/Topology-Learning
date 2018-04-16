"""
This script executes the task of estimating the building type, based solely on the geometry signature for that
building. The data for this script is committed in the repository and can be regenerated by running the prep/get-data.sh
and prep/preprocess-buildings.py scripts, which will take about an hour or two.

This script itself will run for about two minutes depending on your hardware, if you have at least a recent i7 or
comparable
"""

import multiprocessing
import os
import sys
from time import time
from datetime import datetime, timedelta

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# https://chrisyeh96.github.io/2017/08/08/definitive-guide-python-imports.html#case-4-importing-from-parent-directory
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from topoml_util.slack_send import notify

SCRIPT_VERSION = '1.0.1'
SCRIPT_NAME = os.path.basename(__file__)
TIMESTAMP = str(datetime.now()).replace(':', '.')
NUM_CPUS = multiprocessing.cpu_count() - 1 or 1
DATA_FOLDER = SCRIPT_DIR + '/../../files/buildings/'
FILENAME_PREFIX = 'buildings_order_30_train'
EFD_ORDERS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24]
SCRIPT_START = time()

if __name__ == '__main__':  # this is to squelch warnings on scikit-learn multithreaded grid search
    # Load training data
    training_files = []
    for file in os.listdir(DATA_FOLDER):
        if file.startswith(FILENAME_PREFIX) and file.endswith('.npz'):
            training_files.append(file)

    train_fourier_descriptors = np.array([])
    train_labels = np.array([])

    for index, file in enumerate(training_files):  # load and concatenate the training files
        train_loaded = np.load(DATA_FOLDER + file)

        if index == 0:
            train_fourier_descriptors = train_loaded['fourier_descriptors']
            train_labels = train_loaded['building_type']
        else:
            train_fourier_descriptors = \
                np.append(train_fourier_descriptors, train_loaded['fourier_descriptors'], axis=0)
            train_labels = \
                np.append(train_labels, train_loaded['building_type'], axis=0)

    scaler = StandardScaler().fit(train_fourier_descriptors)
    train_fourier_descriptors = scaler.transform(train_fourier_descriptors)

    param_grid = {'max_depth': range(6, 13)}

    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(
        DecisionTreeClassifier(),
        n_jobs=NUM_CPUS,
        param_grid=param_grid,
        verbose=1,
        cv=cv)

    print('Performing grid search on model...')
    print('Using {} threads for grid search'.format(NUM_CPUS))
    print('Searching {} elliptic fourier descriptor orders'.format(EFD_ORDERS))

    best_order = 0
    best_score = 0
    best_params = {}

    for order in EFD_ORDERS:
        print('\nFitting order {} fourier descriptors'.format(order))
        stop_position = 3 + (order * 8)
        grid.fit(train_fourier_descriptors[:, :stop_position], train_labels)
        print("The best parameters for order {} are {} with a score of {}\n".format(
            order, grid.best_params_, grid.best_score_))
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_order = order
            best_params = grid.best_params_

    print('\Training model on order {} with best parameters {}'.format(
        best_order, best_params))
    stop_position = 3 + (best_order * 8)
    clf = DecisionTreeClassifier(max_depth=best_params['max_depth'])
    scores = cross_val_score(clf, train_fourier_descriptors[:, :stop_position], train_labels, cv=10, n_jobs=NUM_CPUS)
    print('Cross-validation scores:', scores)
    clf.fit(train_fourier_descriptors[:, :stop_position], train_labels)

    # Run predictions on unseen test data to verify generalization
    TEST_DATA_FILE = DATA_FOLDER + 'buildings_order_30_test.npz'
    test_loaded = np.load(TEST_DATA_FILE)
    test_fourier_descriptors = test_loaded['fourier_descriptors']
    test_labels = np.asarray(test_loaded['building_type'], dtype=int)
    test_fourier_descriptors = scaler.transform(test_fourier_descriptors)

    print('Run on test data...')
    predictions = clf.predict(test_fourier_descriptors[:, :stop_position])
    test_accuracy = accuracy_score(test_labels, predictions)

    runtime = time() - SCRIPT_START
    message = '\nTest accuracy of {} for fourier descriptor order {} with {} in {}'.format(
        test_accuracy, best_order, best_params, timedelta(seconds=runtime))
    print(message)
    notify(SCRIPT_NAME, message)
    print(SCRIPT_NAME, 'finished successfully in {}'.format(timedelta(seconds=runtime)))
