import numpy as np

from si.base.model import Model
from si.metrics import rmse


class KNNRegressor(Model):

    def __init__(self, k=3, distance=None, **kwargs):
        super().__init__(**kwargs)

        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset):
        self.dataset = dataset
        return self

    def _predict(self, dataset):
        predictions = []

        for sample in dataset.X:

            distances = []

            for train_sample in self.dataset.X:
                d = self.distance(sample, train_sample)
                distances.append(d)

            indexes = np.argsort(distances)[:self.k]

            nearest_values = self.dataset.y[indexes]

            prediction = np.mean(nearest_values)

            predictions.append(prediction)

        return np.array(predictions)

    def _score(self, dataset):
        y_pred = self._predict(dataset)

        return rmse(dataset.y, y_pred)