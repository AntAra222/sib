
from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):

    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def derivative(self, y_true, y_pred):
        pass


class MeanSquaredError(LossFunction):

    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        n = len(y_true)
        return (2 / n) * (y_pred - y_true)


class BinaryCrossEntropy(LossFunction):

    def loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return -np.mean(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        )

    def derivative(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        n = len(y_true)

        return (1 / n) * (
            (y_pred - y_true) /
            (y_pred * (1 - y_pred))
        )
