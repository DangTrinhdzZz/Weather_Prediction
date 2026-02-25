"""
Reducing the number of input variables after feature engineering
- Unsupervised Methods: Ignore the target variable ( removing 
features with low variance or high correlation)
- Supervised Methods (using feature importance from Random Forest)
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
import warnings
warnings.filterwarnings('ignore')



class DropHighlyCorrelated1(BaseEstimator, TransformerMixin):
    """
    Drop feature khi có tương quan cao hơn threshold, dùng phương pháp loại dần.
    """
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        numeric_df = X.select_dtypes(include=['number']).copy()

        # Nếu có target "temp" lẫn trong X, bỏ ra
        if 'temp' in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=['temp'])

        corr_matrix = numeric_df.corr().abs()
        # loại bỏ self-corr (đường chéo)
        np.fill_diagonal(corr_matrix.values, 0)

        # Loại bỏ từng feature một cách an toàn
        to_drop = set()
        cols = corr_matrix.columns

        for col in cols:
            # Bỏ qua nếu col đã bị drop
            if col in to_drop:
                continue

            # Tìm các feature tương quan cao với col
            high_corr = corr_matrix.index[corr_matrix[col] > self.threshold].tolist()
            # Loại tất cả các feature còn lại có corr cao hơn threshold với col
            to_drop.update(high_corr)

        self.to_drop_ = list(to_drop)
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.to_drop_, errors='ignore')

    
class FeatureSelectionLGB(BaseEstimator, TransformerMixin):
    """
    Feature selection using LightGBM feature importances (averaged over multiple targets).
    """

    def __init__(self, top_k=30, random_state=42, importance_type='gain'):
        self.top_k = top_k
        self.random_state = random_state
        self.importance_type = importance_type
        self.selected_features_ = None

    def fit(self, X, y):
        base_model = LGBMRegressor(
            random_state=self.random_state,
            importance_type=self.importance_type,
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31
        )

        # Bọc MultiOutput
        multi_model = MultiOutputRegressor(base_model, n_jobs=-1)
        multi_model.fit(X, y)

        # Lấy importance từ từng estimator
        all_importances = []
        for est in multi_model.estimators_:
            all_importances.append(est.feature_importances_)

        # Trung bình importance qua tất cả đầu ra
        mean_importances = np.mean(all_importances, axis=0)

        # Chọn top_k feature
        importance_series = pd.Series(mean_importances, index=X.columns)
        self.selected_features_ = importance_series.nlargest(self.top_k).index.tolist()

        return self

    def transform(self, X, y=None):
        if self.selected_features_ is None:
            raise RuntimeError("You must fit before calling transform().")
        # Lọc các cột được chọn
        return X[self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return self.selected_features_
class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Flexible Feature Selection supporting Wrapper & Embedded methods.
    Supported methods:
        - 'rfe': Recursive Feature Elimination
        - 'forward': Sequential Forward Selection
        - 'backward': Sequential Backward Selection
        - 'lasso': L1 regularization
        - 'ridge': L2 regularization
        - 'elasticnet': Combination of L1 & L2
    """

    def __init__(self, method="rfe", top_k=30, random_state=42, alpha=1.0, l1_ratio=0.5):
        self.method = method
        self.top_k = top_k
        self.random_state = random_state
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.selected_features_ = None

    def fit(self, X, y):
        X = pd.DataFrame(X)  # đảm bảo có cột tên khi lấy feature importance

        # ===== Wrapper Methods =====
        if self.method == "rfe":
            base_model = LinearRegression()
            selector = RFE(base_model, n_features_to_select=self.top_k)
            selector.fit(X, y)
            self.selected_features_ = X.columns[selector.support_].tolist()

        elif self.method in ["forward", "backward"]:
            base_model = LinearRegression()
            direction = "forward" if self.method == "forward" else "backward"
            selector = SequentialFeatureSelector(base_model,
                                                 n_features_to_select=self.top_k,
                                                 direction=direction,
                                                 n_jobs=-1)
            selector.fit(X, y)
            self.selected_features_ = X.columns[selector.get_support()].tolist()

        # ===== Embedded Methods =====
        elif self.method == "lasso":
            model = Lasso(alpha=self.alpha, random_state=self.random_state)
            model.fit(X, y)
            importance = np.abs(model.coef_)
            self.selected_features_ = X.columns[np.argsort(importance)[-self.top_k:]].tolist()

        elif self.method == "ridge":
            model = Ridge(alpha=self.alpha, random_state=self.random_state)
            model.fit(X, y)
            importance = np.abs(model.coef_)
            self.selected_features_ = X.columns[np.argsort(importance)[-self.top_k:]].tolist()

        elif self.method == "elasticnet":
            model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=self.random_state)
            model.fit(X, y)
            importance = np.abs(model.coef_)
            self.selected_features_ = X.columns[np.argsort(importance)[-self.top_k:]].tolist()

        else:
            raise ValueError(f"Unknown method '{self.method}'")

        return self

    def transform(self, X, y=None):
        if self.selected_features_ is None:
            raise RuntimeError("You must fit before calling transform().")
        return X[self.selected_features_]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
class FeatureSelector1(BaseEstimator, TransformerMixin):
    """
    Flexible Feature Selection supporting Wrapper & Embedded methods.
    Supports multi-output y (multiple target columns).
    
    Supported methods:
        - 'rfe': Recursive Feature Elimination
        - 'forward': Sequential Forward Selection
        - 'backward': Sequential Backward Selection
        - 'lasso': L1 regularization
        - 'ridge': L2 regularization
        - 'elasticnet': Combination of L1 & L2
    """

    def __init__(self, method="rfe", top_k=30, random_state=42, alpha=1.0, l1_ratio=0.5):
        self.method = method
        self.top_k = top_k
        self.random_state = random_state
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.selected_features_ = None

    def fit(self, X, y):
        X = pd.DataFrame(X)

        # Wrapper Methods
        if self.method == "rfe":
            base_model = LinearRegression()
            selector = RFE(base_model, n_features_to_select=self.top_k)
            selector.fit(X, y)
            self.selected_features_ = X.columns[selector.support_].tolist()

        elif self.method in ["forward", "backward"]:
            base_model = LinearRegression()
            direction = "forward" if self.method == "forward" else "backward"
            selector = SequentialFeatureSelector(
                base_model,
                n_features_to_select=self.top_k,
                direction=direction,
                n_jobs=-1
            )
            selector.fit(X, y)
            self.selected_features_ = X.columns[selector.get_support()].tolist()

        # Embedded Methods (support multi-output)
        elif self.method == "lasso":
            model = Lasso(alpha=self.alpha, random_state=self.random_state)
            model.fit(X, y)
            importance = np.abs(model.coef_)
            if importance.ndim > 1:  # multi-output case
                importance = np.mean(importance, axis=0)
            self.selected_features_ = X.columns[np.argsort(importance)[-self.top_k:]].tolist()

        elif self.method == "ridge":
            model = Ridge(alpha=self.alpha, random_state=self.random_state)
            model.fit(X, y)
            importance = np.abs(model.coef_)
            if importance.ndim > 1:
                importance = np.mean(importance, axis=0)
            self.selected_features_ = X.columns[np.argsort(importance)[-self.top_k:]].tolist()

        elif self.method == "elasticnet":
            model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, random_state=self.random_state)
            model.fit(X, y)
            importance = np.abs(model.coef_)
            if importance.ndim > 1:
                importance = np.mean(importance, axis=0)
            self.selected_features_ = X.columns[np.argsort(importance)[-self.top_k:]].tolist()

        else:
            raise ValueError(f"Unknown method '{self.method}'")

        return self

    def transform(self, X, y=None):
        if self.selected_features_ is None:
            raise RuntimeError("You must fit before calling transform().")
        return X[self.selected_features_]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
