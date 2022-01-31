from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np

class SelectColumns(TransformerMixin):
    def __init__(self, cols=[]):
        self.cols = cols
    def fit(self, X=None, y=None, **fit_params):
        return self
    def transform(self, data):
        X = data.copy()
        X = X[self.cols]
        return X