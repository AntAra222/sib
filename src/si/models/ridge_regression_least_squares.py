import numpy as np

from si.base.model import Model
from si.metrics import mse


class RidgeRegressionLeastSquares(Model):

    def __init__(self, l2_penalty=0.01, scale=True, **kwargs):
        super().__init__(**kwargs)

        # Parameters
        self.l2_penalty = l2_penalty
        self.scale = scale

        # Estimated parameters
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _fit(self, dataset):
        X = np.array(dataset.X, dtype=float)
        y = np.array(dataset.y, dtype=float)

        # 1. Scale the data if required
        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)

            # Avoid division by zero
            self.std[self.std == 0] = 1

            X = (X - self.mean) / self.std

        else:
            self.mean = np.zeros(X.shape[1])
            self.std = np.ones(X.shape[1])

        # 2. Add intercept term to X
        X = np.c_[np.ones(X.shape[0]), X]

        # 3. Compute penalty matrix
        penalty = self.l2_penalty * np.eye(X.shape[1])

        # 4. Do not penalize the intercept
        penalty[0, 0] = 0

        # 5. Compute model parameters
        theta_all = np.linalg.inv(
            X.T.dot(X) + penalty
        ).dot(
            X.T.dot(y)
        )

        # First element is theta_zero
        self.theta_zero = theta_all[0]

        # Remaining elements are theta
        self.theta = theta_all[1:]

        return self

    def _predict(self, dataset):
        X = np.array(dataset.X, dtype=float)

        # 1. Scale using mean and std estimated during fit
        if self.scale:
            X = (X - self.mean) / self.std

        # 2. Add intercept term
        X = np.c_[np.ones(X.shape[0]), X]

        # 3. Concatenate theta_zero and theta
        thetas = np.r_[
            self.theta_zero,
            self.theta
        ]

        # 4. Calculate predictions
        y_pred = X.dot(thetas)

        return y_pred

    def _score(self, dataset):
        # 1. Get predictions
        y_pred = self._predict(dataset)

        # 2. Calculate MSE
        return mse(dataset.y, y_pred)