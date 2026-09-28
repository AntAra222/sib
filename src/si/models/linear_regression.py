import numpy as np

from si.base.model import Model
from si.metrics import mse


class RidgeRegression(Model):

    def __init__(
        self,
        l2_penalty=0.01,
        alpha=0.01,
        max_iter=1000,
        patience=10,
        scale=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        # Estimated parameters
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

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

        # Initialize parameters
        self.theta = np.zeros(X.shape[1])
        self.theta_zero = 0.0

        self.cost_history = {}

        best_cost = float("inf")
        no_improvement = 0

        n = len(X)

        for iteration in range(self.max_iter):

            # 2. Predict the values of Y
            y_pred = self.theta_zero + X @ self.theta

            # Calculate the error
            error = y_pred - y

            # 3. Compute the gradient for theta
            gradient_theta = (X.T @ error) / n

            # 4. Compute the gradient for L2 regularization
            regularization_gradient = self.l2_penalty * self.theta

            gradient_theta += regularization_gradient

            # 5. Update theta
            self.theta -= self.alpha * gradient_theta

            # 6. Update theta_zero
            gradient_theta_zero = np.mean(error)

            self.theta_zero -= self.alpha * gradient_theta_zero

            # 7. Compute the cost function
            current_cost = self.cost(dataset)

            self.cost_history[iteration] = current_cost

            # 8. Stop if there is no improvement
            if current_cost < best_cost:
                best_cost = current_cost
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= self.patience:
                break

        return self

    def _predict(self, dataset):

        X = np.array(dataset.X, dtype=float)

        # Apply the same scaling used during training
        if self.scale:
            X = (X - self.mean) / self.std

        # 1. Predict Y using theta and theta_zero
        y_pred = self.theta_zero + X @ self.theta

        return y_pred

    def _score(self, dataset):

        # 1. Predict Y
        y_pred = self._predict(dataset)

        # 2. Calculate MSE
        return mse(dataset.y, y_pred)

    def cost(self, dataset):

        # 1. Predict Y
        y_pred = self._predict(dataset)

        y_true = np.array(dataset.y, dtype=float)

        # 2. Calculate the cost function
        error = np.mean((y_true - y_pred) ** 2)

        regularization = self.l2_penalty * np.sum(self.theta ** 2)

        return error + regularization