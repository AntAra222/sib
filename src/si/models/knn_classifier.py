import numpy as np

from si.base.model import Model
from si.metrics import accuracy


class KNNClassifier(Model):

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

            nearest_classes = self.dataset.y[indexes]

            classes, counts = np.unique(
                nearest_classes,
                return_counts=True
            )

            prediction = classes[np.argmax(counts)]

            predictions.append(prediction)

        return np.array(predictions)

    def _score(self, dataset):
        y_pred = self._predict(dataset)
    
        return accuracy(dataset.y, y_pred)