"""
Data Preprocessing for Hanoi Temperature Forecasting
- Load raw CSV
- Handle missing values, numeric/categorical coercion
...
- Save processed CSV
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


# 1. Cho toàn bộ data set
def load_data(input_path: str):
    """
    Load dataset từ đường dẫn CSV.
    """
    try:
        df = pd.read_csv(input_path)
        print(f" Loaded data with shape: {df.shape}")
        return df.drop_duplicates()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {input_path}")


def transform_dtype(df: pd.DataFrame):
    """
    Chuyển đổi các kiểu dữ liệu phù hợp.
    """
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values("datetime").set_index("datetime")
    for col in ["sunrise", "sunset"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def drop_redundant_column(df: pd.DataFrame):
    """
    Drop description (trùng thông tin với conditions), 
    icon (trùng thông tin với các thông số khác), 
    severisk sẽ drop thông qua handle missing khi miss quá 5% (feature này chỉ available từ năm 2023)
    stations cho thấy insignificant khi kiểm tra ở data_understanding
    name chỉ có 1 location Hanoi
    
    Parameters:
    - df: DataFrame cần xử lý (entire data)

    Returns:
    - df: DataFrame sau khi đã drop column
    """
    # Drop 'description' nếu có
    if 'description' in df.columns:
        df = df.drop('description', axis=1)
        print("Dropped column: 'description'")
    else:
        print("Column 'description' not found, skip dropping.")

    if 'icon' in df.columns:
        df = df.drop('icon', axis=1)
        print("Dropped column: 'icon'")
    else:
        print("Column 'icon' not found, skip dropping.")

    if 'stations' in df.columns:
        df = df.drop('stations', axis=1)
        print("Dropped column: 'stations'")
    else:
        print("Column 'station' not found, skip dropping.")

    if 'name' in df.columns:
        df = df.drop('name', axis=1)
        print("Dropped column: 'name'")

    else:
        print("Column 'name' not found, skip dropping.")
        
    return df.drop_duplicates()


def basic_preprocessing(df: pd.DataFrame):
    """
    Chạy các bước preprocessing không phụ thuộc train/test
    """
    df = transform_dtype(df)
    df = drop_redundant_column(df)
    return df

def drop_redundant_column_hourly(df: pd.DataFrame):
    """
    Drops features deemed redundant or insignificant based on the initial data analysis.

    Columns and reasons for dropping:
    - 'description': Redundant information, often duplicating data found in 'conditions'.
    - 'icon': Redundant visual information, covered by 'conditions' and other parameters.
    - 'stations': Insignificant feature identified during data understanding.
    - 'precipprob': Precipitation probability, information can often be inferred from 'preciptype' and 'conditions'.
    - 'preciptype': Type of precipitation, information can often be inferred from 'conditions'.
    - 'source': Source information is irrelevant for the modeling task.
    - 'severerisk': Will be dropped later (via missing data handling) as it's missing >5% of data (only available since 2023).
    - ('name', 'address', 'resolvedAddress'): All three are redundant as the entire dataset only contains data for a single location ('Hanoi').
    - ('snow', 'snowdepth'): Highly insignificant as snow accumulation is typically zero in this geographical region (Hanoi).
    
    Parameters:
    - df: The input DataFrame to process.

    Returns:
    - df: The DataFrame after dropping the unnecessary columns.
    """
    # List of all columns to check and potentially drop
    columns_to_drop = [
        'description', 
        'icon', 
        'stations', 
        'name', 
        'address', 
        'resolvedAddress', 
        'precipprob', 
        'preciptype', 
        'severerisk', 
        'source', 
        'snow', 
        'snowdepth',
        'longitude',
        'latitude'
    ]

    print("--- Starting column dropping process ---")

    # Iterate through the list of columns
    for column in columns_to_drop:
        if column in df.columns:
            # Drop the column if it exists
            df = df.drop(column, axis=1)
            print(f"Dropped column: '{column}'")
        else:
            # Skip if the column is not found
            print(f"Column not found: '{column}', skipping.")

    print("--- Column dropping process finished ---")

    return df.drop_duplicates()

def basic_preprocessing_hourly(df: pd.DataFrame):
    """
    Chạy các bước preprocessing không phụ thuộc train/test
    """
    df = transform_dtype(df)
    df = drop_redundant_column_hourly(df)
    return df


 
# mục đích loại cột missing preciptype
class HandleMissing(BaseEstimator, TransformerMixin):
    """Drop và fill missing values"""
    def __init__(self, drop_threshold=0.05):
        self.drop_threshold = drop_threshold
        self.cols_to_drop_ = []
        self.fill_values_ = {}

    def fit(self, X, y=None):
        # Tính tỷ lệ missing
        missing_ratio = X.isnull().mean()
        self.cols_to_drop_ = missing_ratio[missing_ratio > self.drop_threshold].index.tolist()

        # Xác định giá trị fill
        X_train_kept = X.drop(columns=self.cols_to_drop_, errors='ignore')
        for col in X_train_kept.columns:
            if X_train_kept[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X_train_kept[col]):
                    self.fill_values_[col] = X_train_kept[col].median()
                else:
                    self.fill_values_[col] = X_train_kept[col].mode()[0]
        return self

    def transform(self, X, y = None):
        X = X.drop(columns=self.cols_to_drop_, errors='ignore')

        for col, val in self.fill_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)

        return X


# loại các cột numeric chỉ có 1 value (snow, snowdepth)
class DropLowVariance(BaseEstimator, TransformerMixin):
    """Drop numeric feature variance thấp"""
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.kept_cols_ = None

    def fit(self, X, y = None): # input là df chưa tách X,y 
        numeric_df = X.select_dtypes(include=['number'])
        if numeric_df.empty:
            self.kept_cols_ = [c for c in X.columns if c != 'stations']
            return self
        
        selector = VarianceThreshold(threshold=self.threshold)
        selector.fit(numeric_df)

        # Chỉ giữ numeric có variance > threshold
        kept_numeric = numeric_df.columns[selector.get_support()].tolist()
        # Chỉ giữ non-numeric trừ 'stations'
        kept_non_numeric = [c for c in X.select_dtypes(exclude=['number']).columns if c != 'stations']

        self.kept_cols_ = kept_numeric + kept_non_numeric
        return self

    def transform(self, X, y = None):
        return X[self.kept_cols_]


# # loại các cột categorical chỉ có 1 value (preciptype)
class DropCategorical(BaseEstimator, TransformerMixin):
    """Drop categorical feature có nunique == 1 hoặc unique ratio >= 0.9 """
    def __init__(self, unique_ratio_threshold=0.9):
        self.unique_ratio_threshold = unique_ratio_threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        cat_cols = X.select_dtypes(include=['object']).columns
        n_rows = len(X)
        self.to_drop_ = []
        for col in cat_cols:
            nunique_ratio = X[col].nunique() / n_rows
            if X[col].nunique() == 1 or nunique_ratio > self.unique_ratio_threshold:
                self.to_drop_.append(col)
        return self

    def transform(self, X, y = None):
        return X.drop(columns=self.to_drop_, errors='ignore')
    
    
class SeasonClassifier(BaseEstimator, TransformerMixin):
    """
    Phân loại mùa theo tháng hoặc tự động dựa trên nhiệt độ trung bình.
    n_season: số nhóm season muốn chia
    is_category = sử dụng biến ở dạng numeric hay category
    """
    def __init__(self, temp_col='temp',n_seasons=5, is_category = True):
        self.temp_col = temp_col
        self.n_seasons = n_seasons
        self.is_category = is_category
        self.season_map_ = None

    def fit(self, X, y=None):
        df = X.copy()
        df['month'] = X.index.month

        # Nếu y là Series chứa nhiệt độ
        if isinstance(y, (pd.Series, np.ndarray)):
            df[self.temp_col] = y.values

        # Tính nhiệt độ trung bình theo tháng
        month_mean = df.groupby('month')[self.temp_col].mean().reset_index()

        # Sắp xếp theo nhiệt độ tăng dần
        month_mean = month_mean.sort_values(self.temp_col).reset_index(drop=True)

        # Chia thành 5 mùa theo quantile
        bins = pd.qcut(month_mean[self.temp_col], q=self.n_seasons,
                       labels=False, duplicates='drop')
        month_mean['season_id'] = bins

        # Lưu map {month → season_id}
        self.season_map_ = dict(zip(month_mean['month'], month_mean['season_id']))
        return self

    def transform(self, X):
        df = X.copy()

        if isinstance(df.index, pd.DatetimeIndex):
            df['month'] = df.index.month
        elif 'month' not in df.columns:
            raise ValueError("Cần có index dạng datetime hoặc cột 'month'.")


        if self.season_map_ is None:
            raise ValueError("Bạn cần gọi .fit() trước khi .transform()")
        

        if self.is_category:  # để là category
            df['season'] = df['month'].map(self.season_map_) 
            df['season'] = df['season'].astype('category')
            return df
        
        df['season'] = df['month'].map(self.season_map_) # dạng numeric int

        return df
    
class WindCategoryEncoder(BaseEstimator, TransformerMixin):
    """
    Chuyển đổi wind category thành numeric encoding dựa trên nhiệt độ trung bình và chia quantile.
    is_category: sử dụng biến ở dạng numeric hay category
    """
    def __init__(self, wind_category_col='wind_category', is_category=False, n_quantiles=4):
        self.wind_category_col = wind_category_col
        self.is_category = is_category
        self.n_quantiles = n_quantiles
        self.encoding_map_ = None
        self.categories_ = None

    def fit(self, X, y=None):
        df = X.copy()

        if self.wind_category_col not in df.columns:
            raise ValueError(f"Column '{self.wind_category_col}' not found in data")

        # Lấy categories từ training data
        self.categories_ = df[self.wind_category_col].cat.categories.tolist()
            
        if y is None:
            raise ValueError("Cần cung cấp y (target values) cho việc encoding")
        
        self._fit_quantile_encoding(df, y)
            
        return self

    def _fit_quantile_encoding(self, df, y):
        """Encoding dựa trên nhiệt độ trung bình và chia quantile"""
        # Tính mean target cho mỗi category
        encoding_df = pd.DataFrame({
            'category': df[self.wind_category_col],
            'target': y
        })
        
        # Tính nhiệt độ trung bình cho từng wind category
        category_means = encoding_df.groupby('category')['target'].mean().reset_index()
        
        # Sắp xếp theo nhiệt độ trung bình tăng dần
        category_means = category_means.sort_values('target').reset_index(drop=True)

        # Chia thành n quantile dựa trên nhiệt độ trung bình
        bins = pd.qcut(category_means['target'], q=self.n_quantiles,
                      labels=False, duplicates='drop')
        category_means['quantile_id'] = bins

        # Lưu map {category → quantile_id}
        self.encoding_map_ = dict(zip(category_means['category'], category_means['quantile_id']))

    def transform(self, X):
        df = X.copy()
        if self.is_category:
            # sử dụng wind_category cũ không biến đổi nữa
            return df

        if self.encoding_map_ is None:
            raise ValueError("Bạn cần gọi .fit() trước khi .transform()")
        
        encoded_col = 'numeric_wind_category' 
        # Áp dụng mapping
        df[encoded_col] = df[self.wind_category_col].map(self.encoding_map_)
        
        # Xử lý unknown categories (nếu có categories mới trong test data)
        unknown_mask = df[encoded_col].isna()
        if unknown_mask.any():
            unknown_cats = df.loc[unknown_mask, self.wind_category_col].unique()
            df.loc[unknown_mask, encoded_col] = -1

        
        # Dạng numeric, drop category cũ
        df[encoded_col] = df[encoded_col].astype(float)
        df = df.drop('wind_category', axis=1)
        
        return df
    
class ConditionsEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, conditions_col='conditions', 
                 encoding_method='target',  # 'ordinal', 'target', 'quantile'
                 n_quantiles=3,
                 is_category=False):
        self.conditions_col = conditions_col
        self.encoding_method = encoding_method
        self.n_quantiles = n_quantiles
        self.is_category = is_category
        self.encoding_map_ = None
        self.binary_maps_ = None

    def fit(self, X, y=None):
        df = X.copy()
        
        if y is None:
            raise ValueError("Cần cung cấp y (target values)")
        
        if self.encoding_method == 'ordinal':
            self._fit_ordinal_encoding(df, y)
        elif self.encoding_method == 'target':
            self._fit_target_encoding(df, y)
        elif self.encoding_method == 'quantile':
            self._fit_quantile_encoding(df, y)
        else:
            raise ValueError("encoding_method không hợp lệ")
            
        return self

    def _fit_ordinal_encoding(self, df, y):
        """Ordinal encoding theo nhiệt độ trung bình"""
        encoding_df = pd.DataFrame({
            'conditions': df[self.conditions_col],
            'target': y
        })
        
        conditions_means = encoding_df.groupby('conditions')['target'].mean()
        conditions_means = conditions_means.sort_values()
        
        self.encoding_map_ = {cond: idx for idx, cond in enumerate(conditions_means.index)}

    def _fit_target_encoding(self, df, y):
        """Target encoding với smoothing cho conditions ít data"""
        encoding_df = pd.DataFrame({
            'conditions': df[self.conditions_col],
            'target': y
        })
        
        conditions_means = encoding_df.groupby('conditions')['target'].mean()
        conditions_counts = df[self.conditions_col].value_counts()
        
        global_mean = y.mean()
        
        self.encoding_map_ = {}
        for condition in conditions_means.index:
            cond_mean = conditions_means[condition]
            cond_count = conditions_counts[condition]
            
            # Smoothing cho conditions ít data (Rain chỉ có 8 samples)
            alpha = max(50, 100 - cond_count)  
            smoothed_mean = (cond_count * cond_mean + alpha * global_mean) / (cond_count + alpha)
            
            self.encoding_map_[condition] = smoothed_mean

    def _fit_quantile_encoding(self, df, y):
        """Chia conditions thành quantile dựa trên nhiệt độ"""
        encoding_df = pd.DataFrame({
            'conditions': df[self.conditions_col],
            'target': y
        })
        
        conditions_means = encoding_df.groupby('conditions')['target'].mean().reset_index()
        conditions_means = conditions_means.sort_values('target')
        
        # Chia quantile
        if len(conditions_means) >= self.n_quantiles:
            bins = pd.qcut(conditions_means['target'], q=self.n_quantiles, 
                          labels=False, duplicates='drop')
            conditions_means['quantile_id'] = bins
        else:
            conditions_means['quantile_id'] = range(len(conditions_means))
        
        self.encoding_map_ = dict(zip(conditions_means['conditions'], conditions_means['quantile_id']))


    def transform(self, X):
        df = X.copy()
        if self.is_category:
            # sử dụng conditions cũ không biến đổi nữa
            return df

        if self.encoding_map_ is None:
            raise ValueError("Bạn cần gọi .fit() trước khi .transform()")
        
        encoded_col = 'numeric_conditions' 
        # Áp dụng mapping
        df[encoded_col] = df[self.conditions_col].map(self.encoding_map_)
        
        # Xử lý unknown categories (nếu có categories mới trong test data)
        unknown_mask = df[encoded_col].isna()
        if unknown_mask.any():
            unknown_cats = df.loc[unknown_mask, self.conditions_col].unique()
            df.loc[unknown_mask, encoded_col] = -1

        
        # Dạng numeric, drop category cũ
        df[encoded_col] = df[encoded_col].astype(float)
        df = df.drop(self.conditions_col, axis=1)

        return df

class To_Category(BaseEstimator, TransformerMixin):
    """
    Chuyển các object thành category cho LightGBM.
    - Chỉ fit trên train
    - Chuyển object/string -> pandas 'category' dtype
    - Không mã hóa số (giữ để LGB xử lý nội bộ)
    """
    def __init__(self):
        self.columns = None
        self.categories_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        self.columns = X.select_dtypes(include=['category', 'object']).columns
        for col in self.columns:
            if col in X.columns:
                # lưu lại các category duy nhất (tránh lỗi unseen category)
                self.categories_[col] = X[col].astype('category').cat.categories
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].astype('category')
                # align categories với lúc train
                if col in self.categories_:
                    X[col] = X[col].cat.set_categories(self.categories_[col])
        return X



