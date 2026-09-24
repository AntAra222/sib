import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset, test_size=0.2, random_state=None):
    np.random.seed(random_state)

    n_samples = len(dataset.X)

    permutation = np.random.permutation(n_samples)

    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test

    test_indices = permutation[:n_test]
    train_indices = permutation[n_test:]

    X_train = dataset.X[train_indices]
    X_test = dataset.X[test_indices]

    if dataset.y is not None:
        y_train = dataset.y[train_indices]
        y_test = dataset.y[test_indices]
    else:
        y_train = None
        y_test = None

    train_dataset = Dataset(X_train, y_train)
    test_dataset = Dataset(X_test, y_test)

    return train_dataset, test_dataset

import numpy as np

from si.data.dataset import Dataset


def stratified_train_test_split(dataset, test_size=0.2, random_state=None):
    np.random.seed(random_state)

    train_indices = []
    test_indices = []

    labels, counts = np.unique(dataset.y, return_counts=True)

    for label, count in zip(labels, counts):

        class_indices = np.where(dataset.y == label)[0]

        class_indices = np.random.permutation(class_indices)

        n_test = int(count * test_size)

        test_indices.extend(class_indices[:n_test])

        train_indices.extend(class_indices[n_test:])

    X_train = dataset.X[train_indices]
    y_train = dataset.y[train_indices]


    X_test = dataset.X[test_indices]
    y_test = dataset.y[test_indices]

    train_dataset = Dataset(X_train, y_train)
    test_dataset = Dataset(X_test, y_test)

    return train_dataset, test_dataset