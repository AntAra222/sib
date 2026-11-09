
from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w, grad_loss_w):
        pass

class SGD(Optimizer):

    def __init__(self, learning_rate, momentum=0.0):
        super().__init__(learning_rate)

        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w, grad_loss_w):

        # Initialize retained gradient if necessary
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros_like(w)

        # Update retained gradient (momentum)
        self.retained_gradient = (
            self.momentum * self.retained_gradient 
            + (1-self.self.momentum) * grad_loss_w
        )

        # Update weights
        new_weights = (
            w - self.learning_rate * self.retained_gradient
        )

        return new_weights
