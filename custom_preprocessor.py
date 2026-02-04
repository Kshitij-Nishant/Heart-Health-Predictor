from sklearn.preprocessing import PowerTransformer
from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np
import pandas as pd

class SelectiveNumericTransformer(BaseEstimator, TransformerMixin):
    """
    - log transform ONLY chol
    - Yeo-Johnson ONLY oldpeak
    - leaves other numeric columns unchanged
    - works safely inside ColumnTransformer
    """

    def fit(self, X, y=None):
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        self.pt_ = PowerTransformer(
            method='yeo-johnson',
            standardize=False
        )
        self.pt_.fit(X[:, [4]])  # oldpeak (5th numeric column)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = X.copy()

        # log transform chol (index 3)
        X[:, 3] = np.log(X[:, 3])

        # Yeo-Johnson transform oldpeak (index 4)
        X[:, 4] = self.pt_.transform(X[:, [4]]).ravel()

        return X