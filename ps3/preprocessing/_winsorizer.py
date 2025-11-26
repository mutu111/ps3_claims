import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile = 0.01, upper_quantile = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=False)

        self.lower_quantile_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_quantile_ = np.quantile(X, self.upper_quantile, axis=0)

        return self
    
    def transform(self, X):
        check_is_fitted(self, ["lower_quantile_", "upper_quantile_"])

        X = check_array(X, accept_sparse=False)

        clip = np.clip(X, self.lower_quantile_, self.upper_quantile_)

        return clip
