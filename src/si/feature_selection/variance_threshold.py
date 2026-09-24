import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset


class VarianceThreshold(Transformer):

    def __init__(self, threshold=0.0, **kwargs):
        super().__init__(**kwargs)

        self.threshold = threshold
        self.variance = None

    def _fit(self, dataset):
        self.variance = np.var(dataset.X, axis=0)
        return self

    def _transform(self, dataset):
        mask = self.variance > self.threshold

        X = dataset.X[:, mask]

        if dataset.y is not None:
            y = dataset.y
        else:
            y = None

        if dataset.features is not None:
            features = np.array(dataset.features)[mask]
        else:
            features = None

        return Dataset(X, y, features)