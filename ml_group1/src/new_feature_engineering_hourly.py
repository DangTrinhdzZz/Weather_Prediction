"""
Feature Engineering
- Time-based features (dayofyear, month, weekday, weekend)
- Lag features: Nhiều biến trong chuỗi thời gian phụ thuộc vào giá trị trước đó của chính nó (auto-correlation).
- rolling mean/std for temp
- Sử dụng category type thay vì one-hot encoding cho categorical features
"""

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

def create_date_features(df, forecast_horizon=5):
    df = df.copy()
    base_dates = df.index
    
    # Date features cho current day (t)
    df['month'] = base_dates.month
    df['weekday'] = base_dates.weekday
    df['day_of_year'] = base_dates.dayofyear
    df['is_weekend'] = (base_dates.weekday >= 5).astype(int)
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Tạo date features cho các horizons t+1 đến t+5
    for horizon in range(1, forecast_horizon + 1):
        future_dates = base_dates + pd.Timedelta(days=horizon)
        
        df[f'month_h{horizon}'] = future_dates.month
        df[f'weekday_h{horizon}'] = future_dates.weekday
        df[f'day_of_year_h{horizon}'] = future_dates.dayofyear
        df[f'is_weekend_h{horizon}'] = (future_dates.weekday >= 5).astype(int)
        df[f'month_sin_h{horizon}'] = np.sin(2*np.pi*df[f'month_h{horizon}']/12)
        df[f'month_cos_h{horizon}'] = np.cos(2*np.pi*df[f'month_h{horizon}']/12)
        df[f'day_sin_h{horizon}'] = np.sin(2 * np.pi * df[f'day_of_year_h{horizon}'] / 365)
        df[f'day_cos_h{horizon}'] = np.cos(2 * np.pi * df[f'day_of_year_h{horizon}'] / 365)

    return df

def create_specific_features(df, is_drop = True):
    """
    Tạo các features đặc thù từ current data
    """
    df = df.copy()
    
    # Temperature features
    df['temp_range'] = df['tempmax'] - df['tempmin']
    df['dew_spread'] = df['temp'] - df['dew']
    # df['temp_dew_interaction'] = df['temp'] * df['dew']
    
    # Weather condition features
    df['humidity_high'] = (df['humidity'] > 80).astype(int)
    df['heavy_rain'] = (df['precip'] > 50).astype(int)
    
    df["temp_humidity_interaction"] = df["temp"] * df["humidity"]
    df["wind_temp_interaction"] = df["winddir"] * df["temp"]
    
    # Atmospheric indices
    df['wind_temp_index'] = df['windspeed'] * df['temp']
    df['pressure_temp_index'] = df['sealevelpressure'] * df['temp']
    df['humidity_cloud_index'] = (df['humidity'] * df['cloudcover']) / 100
    df['solar_temp_index'] = df['solarradiation'] * df['temp']
    df['uv_cloud_index'] = 1 - df['cloudcover'] / 100
    df['solar_visibility_index'] = df['visibility'] * (1 - df['cloudcover'] / 100)
    df['wind_variability'] = df['windgust'] - df['windspeed']


    # Heat Index (cảm giác nóng, chỉ dùng khi temp ≥ 26°C) ---
    def calculate_heat_index_celsius(temp_c, humidity):
        # Chuyển sang Fahrenheit để dùng công thức gốc
        T = temp_c * 9/5 + 32
        HI_f = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (humidity * 0.094))
        # Đổi lại sang °C
        HI_c = (HI_f - 32) * 5/9
        return HI_c

    # Wind Chill (cảm giác lạnh, chỉ có ý nghĩa khi temp ≤ 10°C) ---
    def calculate_wind_chill(temp_c, windspeed_kmh):
        # Áp dụng công thức Canada (temp bằng °C, windspeed bằng km/h)
        return 13.12 + 0.6215*temp_c - 11.37*(windspeed_kmh**0.16) + 0.3965*temp_c*(windspeed_kmh**0.16)

    # Tính riêng từng chỉ số 
    df['heat_index'] = df.apply(
        lambda r: calculate_heat_index_celsius(r['temp'], r['humidity']) if r['temp'] >= 26 else np.nan,
        axis=1
    )

    df['wind_chill'] = df.apply(
        lambda r: calculate_wind_chill(r['temp'], r['windspeed']) if r['temp'] <= 10 else np.nan,
        axis=1
    )

    # Tạo chỉ số tổng hợp: thermal_index ---
    # Ưu tiên: nhiệt độ nóng → dùng heat_index, lạnh → dùng wind_chill, còn lại → nhiệt độ thật
    df['thermal_index'] = np.where(
        df['temp'] >= 26, df['heat_index'],
        np.where(df['temp'] <= 10, df['wind_chill'], df['temp'])
    )

    
    # Wind direction categorization    
    df["winddir_cos"] = np.cos(np.deg2rad(df["winddir"]))
    df["winddir_sin"] = np.sin(np.deg2rad(df["winddir"]))
    
    def categorize_wind_direction(degree):
        if pd.isna(degree):
            return 'Unknown'
        elif 0 <= degree < 45:
            return 'Bắc_Đông Bắc_N_NE)'  # Thu - Đông
        elif 45 <= degree < 90:
            return 'Đông_Bắc_NE'          # Đông - Xuân
        elif 90 <= degree < 135:
            return 'Đông_Nam_SE'          # Xuân - Hè (Gió Nồm)
        elif 135 <= degree < 180:
            return 'Nam_S'
        elif 180 <= degree < 225:
            return 'Tây_Nam_SW'           # Hè - Thu
        elif 225 <= degree < 270:
            return 'Tây_W'                # Gió Lào
        elif 270 <= degree < 315:
            return 'Tây_Bắc_NW'
        elif 315 <= degree <= 360:
            return 'Bắc_N'
        else:
            return 'Unknown'
        
    # THAY ĐỔI: Ép kiểu về category 
    df['wind_category'] = df['winddir'].apply(categorize_wind_direction).astype('category')
      
    # Season
    # THAY ĐỔI: Ép kiểu về category thay vì one-hot encoding
    df['season'] = df['month'].apply(
        lambda x: 'winter' if x in [12, 1, 2]      # Gió mùa Đông Bắc, temp thấp
        else 'spring' if x in [3, 4]               # Nồm ẩm, temp tăng dần
        else 'summer' if x in [5, 6, 7, 8]         # Oi bức, gió Tây Nam / Tây
        else 'autumn'                              # 9, 10, 11 — mát dần, gió Bắc
    ).astype('category')

    # Weather phenomena
    df['humid_foggy_day'] = (df['precip'] < 1).astype(int)
    df['moisture_index'] = df['humidity'] * (1 - df['precip']/50)
    df['vis_humidity_index'] = df['visibility'] * (1 - df['humidity'] / 100)
    
    return df


def auto_create_lag_features(df):
    """
    Tự động tạo lag features cụ thể cho từng biến, 
    sau đó nối tất cả bằng pd.concat để tối ưu hiệu năng.
    """
    df = df.copy()

    # Cấu hình biến và độ trễ tương ứng
    lag_config = {
        'dew': [1, 2, 3, 7],
        'humidity': [1, 2, 3, 7],
        'precip': [1, 2, 3, 7],
        'windgust': [1, 2, 3, 7],
        'windspeed': [1, 3, 7, 30],
        'winddir_cos': [1, 3, 7],
        'winddir_sin': [1, 3, 7],
        'cloudcover': [1, 2, 3, 7],
        'visibility': [1, 2, 3, 7],
        'sealevelpressure': [1, 2, 3],
        'solarradiation': [1, 2, 3],
        'solarenergy': [1, 2, 3],
        'uvindex': [1, 2, 3],
        'tempmax': [1, 2, 3],
        'tempmin': [1, 2, 3],
        'temp': [1, 2, 3],
        'feelslikemax': [1, 2, 3],
        'feelslikemin': [1, 2, 3],
        'feelslike': [1, 2, 3],
        'temp_range': [1, 2, 3],
        'dew_spread': [1, 2, 3],
        'wind_variability': [1, 2, 3],
        'thermal_index': [1, 2, 3]
    }

    lag_dataframes = []  # chứa các series lag

    for feature, lags in lag_config.items():
        if feature not in df.columns:
            print(f"Bỏ qua'{feature}' (không có trong dataframe)")
            continue
        for lag in lags:
            lag_col_name = f"{feature}_lag_{lag}"
            lag_series = df[feature].shift(lag)
            lag_series.name = lag_col_name
            lag_dataframes.append(lag_series)

    # Ghép tất cả lag features lại cùng lúc
    if lag_dataframes:
        lag_df = pd.concat(lag_dataframes, axis=1)
        df = pd.concat([df, lag_df], axis=1)

    print(f"Đã tạo tổng cộng {len(lag_dataframes)} lag features.")

    return df


def create_rolling_features(df):
    """
    Tạo cụ thể các rolling features theo mapping chi tiết.
    """
    df = df.copy()
    roll_dataframes = []

    # --- 1. Mapping chi tiết: feature -> loại và window cần tạo ---
    rolling_mapping = {
        'dew': {'mean': [7, 14], 'std': [7, 14]},
        'precip': {'mean': [7, 14], 'std': [7, 14], 'sum': [14]},
        'windgust': {'mean': [3, 7], 'std': [3, 7]},
        'windspeed': {'mean': [7], 'std': [7]},
        'winddir_sin': {'mean': [7], 'std': [7]},
        'winddir_cos': {'mean': [7], 'std': [7]},
        'cloudcover': {'mean': [3, 7], 'std': [7, 14]},
        'visibility': {'mean': [3, 7], 'std': [3, 7]},
        'solarradiation': {'mean': [3, 5], 'std': [3, 5], 'sum': [3]},
        'uvindex': {'mean': [3, 4, 5], 'std': [3, 4, 5]},
        'sealevelpressure': {'mean': [3, 5, 7], 'std': [3, 5, 7]},
        'tempmax': {'mean': [3, 5,7], 'std': [3, 5]},
        'tempmin': {'mean': [3, 5], 'std': [3, 5]},
        'temp': {'mean': [3, 5], 'std': [3, 5]},
        'feelslike': {'mean': [3, 5]},
        'feelslikemax': {'mean': [3, 5]},
        'feelslikemin': {'mean': [3, 5]},
        'humidity': {'mean': [7, 14], 'std': [7, 14]},
        'thermal_index': {'mean': [3,21,30]},
        'wind_variability': {'mean': [3, 5], 'std': [3]},
        'dew_spread': {'mean': [3, 5], 'std': [3]},
        'temp_range': {'mean': [3, 5], 'std': [3]},
        'rain_intensity': {'mean': [3, 5]}
    }

    # --- 2. Tạo từng feature theo mapping ---
    for feature, roll_types in rolling_mapping.items():
        if feature not in df.columns:
            print(f"Cảnh báo: Cột '{feature}' không tồn tại trong DataFrame, bỏ qua.")
            continue

        base_series = df[feature].shift(1)  # tránh data leakage

        for roll_type, windows in roll_types.items():
            for window in windows:
                col_name = f"{feature}_roll_{roll_type}_{window}"

                # Tính rolling cụ thể theo loại
                if roll_type == 'mean':
                    new_series = base_series.rolling(window, min_periods=window).mean()
                elif roll_type == 'std':
                    new_series = base_series.rolling(window, min_periods=window).std()
                elif roll_type == 'sum':
                    new_series = base_series.rolling(window, min_periods=window).sum()
                else:
                    continue  # nếu loại chưa hỗ trợ

                new_series.name = col_name
                roll_dataframes.append(new_series)

    # --- 3. Ghép tất cả rolling features ---
    if roll_dataframes:
        roll_df = pd.concat(roll_dataframes, axis=1)
        df = pd.concat([df, roll_df], axis=1)
    
    df = df.drop(['heat_index','wind_chill'], axis = 1)
    print(f"Đã tạo {len(roll_dataframes)} rolling features.")

    return df


def drop_base_features(df):
    """
    Loại bỏ các base feature, chỉ giữ lại derive
    """
    base = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin',
       'feelslike', 'dew', 'humidity',
       'preciptype', 'windgust', 'windspeed', 'winddir',
       'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation',
       'solarenergy', 'uvindex', 'severerisk']
    
    # vẫn giữ condition và icon
    
    return df.drop(base, axis=1)

def feature_engineering(df, forecast_horizon=5, is_drop_nan = False, is_drop_base = False):
    """
    Thực hiện toàn bộ quy trình Feature Engineering cho bài toán dự báo đa đầu ra.
    ĐÃ SỬA: Sử dụng category type thay vì one-hot encoding
    """
    
    # Tạo bản copy để tránh fragmentation
    df = df.copy()
    
    # 1. Tạo targets
    target_cols = []
    target_dataframes = []
    
    for i in range(1, forecast_horizon + 1):
        target_col = f"temp_next_{i}"
        target_series = df['temp'].shift(-i)
        target_series.name = target_col
        target_dataframes.append(target_series)
        target_cols.append(target_col)
    
    # Ghép target columns cùng lúc
    if target_dataframes:
        target_df = pd.concat(target_dataframes, axis=1)
        df = pd.concat([df, target_df], axis=1)

    
    # 2. Tạo date features (được phép dùng vì là thông tin có sẵn)
    df = create_date_features(df)
    
    # 3. Tạo specific features từ current data
    df = create_specific_features(df, is_drop= is_drop_nan)
    
    # 4. CHỌN LỌC features phù hợp cho lag/rolling
    target_features = ['temp_next_1','temp_next_2','temp_next_3','temp_next_4','temp_next_5']

    # 5. Tạo lag features 
    df = auto_create_lag_features(df)
    
    # 6. Tạo rolling features 
    df = create_rolling_features(df)

    # 7. Dropping columns and rows
    df = df.dropna(subset=target_cols)

    if is_drop_nan:
        df = df.dropna()

    if is_drop_base:
        df = drop_base_features(df)
    

    # Tạo defragmented frame
    df = df.copy()

    return df,  target_features