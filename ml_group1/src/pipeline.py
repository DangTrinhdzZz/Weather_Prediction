# pipeline_preprocessing.py
from sklearn.pipeline import Pipeline
from data_preprocessing import HandleMissing, DropLowVariance, DropCategorical, To_Category, ConditionsEncoder, WindCategoryEncoder, SeasonClassifier
from feature_selection import *

def build_preprocessing_pipeline():
    """
    Xây dựng pipeline xử lý dữ liệu: 
    - Chỉ fit trên tập train
    - Gồm: 
        1. Xử lý missing
        3. Drop low variance
        5. Drop categorical ít ý nghĩa
        6. Encode categorical
    """
    pipeline = Pipeline(steps=[
        ("missing", HandleMissing(drop_threshold=0.05)),
        ("drop_low_var", DropLowVariance(threshold=0.0)),
        ("drop_cate", DropCategorical(unique_ratio_threshold=0.9)),
        ("condition_encode", To_Category())
    ])
    return pipeline


def build_encoding_pipeline(is_category=False, encoding_method_condition='target', n_seasons=5, n_quantiles = 4):
    """
    Encoding feature
    1. ConditionsEncoder: 
        Nếu is_category = True, để nguyên conditions
        Else, encode sang numeric theo một trong các cách ['target','ordinal','quantilte'] 

    2. SeasonClassifier : 
        Tạo nhóm season dựa trên nhiệt độ trung bình các tháng, rồi chia thành n_seasons. 
        Nếu is_category = True thì chuyển season.dtype thành category 
        Else, để nguyên kết quả sau khi chia thành n_seasons

    3. WindCategoryEncoder: 
        Nếu is_category = True thì để nguyên wind_category (biến này được tạo từ winddir, phân loại thành 8 loại gió theo domain knowledge trong bước feature engineering
        Else, thì encode sang numeric theo n_qunatiles
    """

    pipeline = Pipeline(steps=[
        ("conditions_encode", ConditionsEncoder(is_category=is_category, encoding_method=encoding_method_condition)),
        ("season_encode", SeasonClassifier(n_seasons=n_seasons, is_category=is_category)),
        ("wind_encode", WindCategoryEncoder(is_category=is_category, n_quantiles= n_quantiles))
    ])

    return pipeline

def build_preprocessing_pipeline_hourly():
    """
    Xây dựng pipeline xử lý dữ liệu:
    - Chỉ fit trên tập train
    - Gồm: 
        1. Xử lý missing
        3. Drop low variance
        5. Drop categorical ít ý nghĩa
        6. Encode categorical
    """
    pipeline = Pipeline(steps=[
        ("missing", HandleMissing(drop_threshold=0.05)),
        ("drop_low_var", DropLowVariance(threshold=0.0)),
        ("drop_cate", DropCategorical(unique_ratio_threshold=0.9))
    ])
    return pipeline

def build_preprocessing_pipeline_hourly():
    """
    Xây dựng pipeline xử lý dữ liệu:
    - Chỉ fit trên tập train
    - Gồm: 
        1. Xử lý missing
        3. Drop low variance
        5. Drop categorical ít ý nghĩa
        6. Encode categorical
    """
    pipeline = Pipeline(steps=[
        ("missing", HandleMissing(drop_threshold=0.05)),
        ("drop_low_var", DropLowVariance(threshold=0.0)),
        ("drop_cate", DropCategorical(unique_ratio_threshold=0.9))
    ])
    return pipeline




