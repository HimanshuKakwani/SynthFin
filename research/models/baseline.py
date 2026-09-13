import numpy as np

class MeanReturnBaseline:
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self
    def predict(self, X):
        return np.full(len(X), self.mean_)
