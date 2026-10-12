
import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function


class LogisticRegression(Model):

    def __init__(self, l2_penalty=0.0, alpha=0.01,
                 max_iter=1000, patience=100, scale=True):

        super().__init__()

        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset: Dataset):

        X = np.asarray(dataset.X, dtype=float)
        y = np.asarray(dataset.y, dtype=float).ravel()

        m, n = X.shape

        # 1. Normalizar os dados, se necessário
        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            self.std[self.std == 0] = 1
            X = (X - self.mean) / self.std
        else:
            self.mean = np.zeros(n)
            self.std = np.ones(n)

        # Inicializar os coeficientes
        self.theta = np.zeros(n)
        self.theta_zero = 0.0
        self.cost_history = {}

        best_cost = np.inf
        no_improvement = 0

        for i in range(self.max_iter):

            # 2. Calcular as probabilidades
            y_pred = sigmoid_function(
                X @ self.theta + self.theta_zero
            )

            # 3. Calcular os gradientes
            error = y_pred - y

            gradient_theta = X.T @ error / m
            gradient_zero = np.sum(error) / m

            # 4. Regularização L2
            regularization = (
                self.l2_penalty / m
            ) * self.theta

            # 5. Atualizar theta
            self.theta -= self.alpha * (
                gradient_theta + regularization
            )

            # 6. Atualizar theta_zero
            self.theta_zero -= self.alpha * gradient_zero

            # 7. Calcular a função de custo
            y_pred = sigmoid_function(
                X @ self.theta + self.theta_zero
            )

            current_cost = self.cost(y, y_pred)
            current_cost += (
                self.l2_penalty / (2 * m)
            ) * np.sum(self.theta ** 2)

            self.cost_history[i] = current_cost

            # 8. Early stopping
            if current_cost < best_cost - 1e-8:
                best_cost = current_cost
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= self.patience:
                break

        return self

    def _predict(self, dataset: Dataset):

        X = np.asarray(dataset.X, dtype=float)

        # Normalizar usando os valores do treino
        if self.scale:
            X = (X - self.mean) / self.std

        # Calcular probabilidades
        y_pred = sigmoid_function(
            X @ self.theta + self.theta_zero
        )

        # Converter probabilidades em 0 e 1
        return (y_pred >= 0.5).astype(int)

    def _score(self, dataset: Dataset):

        # Obter previsões
        y_pred = self._predict(dataset)

        # Calcular accuracy
        return accuracy(dataset.y, y_pred)

    def cost(self, y, y_pred):

        # Evitar log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Binary cross-entropy
        J = -np.mean(
            y * np.log(y_pred)
            + (1 - y) * np.log(1 - y_pred)
        )

        return J
