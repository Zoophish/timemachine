import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ZeroCentreScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scale_ = None

    def fit(self, X, y=None):
        # Find the maximum absolute value in the data
        self.scale_ = np.max(np.abs(X))
        return self

    def transform(self, X):
        if self.scale_ is None:
            raise ValueError("The scaler has not been fitted yet.")
        # Scale the data by dividing by the maximum absolute value
        return X / self.scale_

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class FixedFactorScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaling_factor=1.0):
        """
        Parameters:
        scaling_factor (float): The factor by which to scale all values in the data.
        """
        self.scaling_factor = scaling_factor

    def fit(self, X, y=None):
        # No fitting necessary, as the scaling factor is predefined
        return self

    def transform(self, X):
        return X * self.scaling_factor

    def fit_transform(self, X, y=None):
        return self.transform(X)
    
    def inverse_transform(self, X):
        return X / self.scaling_factor