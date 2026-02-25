import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import joblib
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import StringIO
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
import schedule  # dùng để chạy theo lịch trình

from lightgbm import LGBMRegressor
from src.data_preprocessing import load_data, basic_preprocessing
from src.pipeline import build_preprocessing_pipeline, build_GB_featture_engineering_pipeline
from src.new_feature_engineering_daily import feature_engineering
from src.model_evaluation import evaluate_multi_output, evaluate

# --- PATHs ---
DATA_PATH = r"data\latest_3_year.csv"
API_KEY = r"642BDT8N8D49CTFJCX8ZWU6RT"
MODEL_PATH = r"models\Current_model.pkl"
LOG_PATH = r"logs/daily_rmse.txt"
PIPE_1 = r"pipelines/preprocessing_pipeline.pkl"
# PIPE_2 = r"pipelines/featureSelection_pipeline.pkl"


# --- Lấy dữ liệu mới nhất ---
def fetch_latest_weather_data(location="Hanoi", days=35): 
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    url = f"{base_url}/{location}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

    params = {
        "unitGroup": "metric",
        "include": "days",
        "key": API_KEY,
        "contentType": "csv"
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        print("Lấy dữ liệu thời tiết thành công.")
        df = pd.read_csv(StringIO(response.text))
        df.to_csv(f"data/Current_Raw_3weeks.csv", index=False)
        return df
    else:
        print(f"Lỗi khi gọi API ({response.status_code}): {response.text}")
        return pd.DataFrame()


# --- Cập nhật dữ liệu 3 năm ---
def update_three_year_data(new_data: pd.DataFrame, data_path=DATA_PATH):
    """
    Cập nhật file current_3_year.csv:
      - Đọc file cũ nếu có
      - Gộp thêm dữ liệu mới
      - Loại bỏ trùng lặp theo cột datetime
      - Giữ lại đúng 3 năm gần nhất (tính từ hôm nay)
    """
    # Đảm bảo có cột datetime
    if "datetime" not in new_data.columns:
        raise ValueError("new_data không có cột 'datetime'")

    # Ép kiểu datetime (cho cả new_data)
    new_data["datetime"] = pd.to_datetime(new_data["datetime"], errors="coerce")

    # Đọc dữ liệu cũ nếu có
    if os.path.exists(data_path):
        old_data = pd.read_csv(data_path)
        if "datetime" not in old_data.columns:
            raise ValueError("old_data không có cột 'datetime'")
        old_data["datetime"] = pd.to_datetime(old_data["datetime"], errors="coerce")

        # Gộp dữ liệu
        combined = pd.concat([old_data, new_data], ignore_index=True)
    else:
        combined = new_data.copy()

    # Loại bỏ dòng có datetime bị lỗi hoặc trùng
    combined.dropna(subset=["datetime"], inplace=True)
    combined.drop_duplicates(subset=["datetime"], keep="last", inplace=True)

    # Giới hạn dữ liệu trong 3 năm gần nhất
    three_years_ago = datetime.today() - timedelta(days=3 * 365)
    combined = combined[combined["datetime"] >= three_years_ago]

    # sắp xếp và lưu lại
    combined.sort_values("datetime", inplace=True)
    combined.to_csv(data_path, index=False)

    # kiểm tra
    print(f"Dữ liệu đã được cập nhật, còn lại {len(combined)} dòng (~3 năm).")
    print(f"Khoảng thời gian: {combined['datetime'].min().date()} → {combined['datetime'].max().date()}")


# --- Chuẩn bị dữ liệu ---
def prepare_data(df):

   # 1. basic preprocessing for all data set
   df = basic_preprocessing(df=df)

   # 3. Pipeline 1: preprocessing
   pipeline1 = joblib.load(PIPE_1)
   df_processed = pipeline1.transform(df)

   # 4. Feature engineering cho input
   df_feat, target_col = feature_engineering(df_processed, is_drop_nan= False, is_drop_base= False)

   # 5. Lây input X
   X_df = df_feat.drop(columns= target_col)
   # y_df = train_feat[target_col]

   # 6. Pipeline 2: GB selection
#    pipeline2 = joblib.load(PIPE_2)
#    proccessed_X = pipeline2.transform(X_df)
   #print(5, proccessed_X.columns)

   X_df.to_csv(f"data/Today_X_input.csv", index=False)
   return X_df

# --- Dự đoán ---
def predict_tomorrow(processed_X):
    model = joblib.load(MODEL_PATH) 
    y_pred_5_days = model.predict(processed_X)
    return y_pred_5_days


# --- Ghi log RMSE ---
# sửa lại dùng hàm của mình + vde cụ thể tí t nêu sau
def log_rmse_daily(pred_path, actual_path):
    """
    So sánh dự đoán trong file realtime_predictions với dữ liệu thật trong current3weeks.
    Tính RMSE cho 5 ngày tiếp theo nếu đủ dữ liệu, nếu thiếu thì lưu None.
    """

    # --- Đọc dữ liệu ---
    pred_df = pd.read_csv(pred_path)
    actual_df = pd.read_csv(actual_path)

    pred_df['date'] = pd.to_datetime(pred_df['date'])
    actual_df['datetime'] = pd.to_datetime(actual_df['datetime'])

    # --- Chuẩn bị thư mục ---
    os.makedirs('logs', exist_ok=True)

    # --- Đọc log cũ an toàn ---
    if os.path.exists(LOG_PATH):
        try:
            all_logs = joblib.load(LOG_PATH)
            if not isinstance(all_logs, list):
                all_logs = []
        except Exception:
            all_logs = []
    else:
        all_logs = []

    # --- Lặp qua từng dòng dự báo ---
    for _, row in pred_df.iterrows():
        base_date = row['date']
        forecast_dates = [base_date + timedelta(days=i) for i in range(1, 6)]
        forecast_values = [row[f'pred_day_{i}'] for i in range(1, 6)]

        actual_values = []
        for d in forecast_dates:
            val = actual_df.loc[actual_df['datetime'].dt.date == d.date(), 'temp']
            actual_values.append(val.values[0] if not val.empty else np.nan)

        if np.any(np.isnan(actual_values)):
            rmse_value = None
            status = "Missing data"
        else:
            y_true = np.array([actual_values])
            y_pred = np.array([forecast_values])
            metrics = evaluate_multi_output(y_true, y_pred)
            rmse_value = metrics["average"]["RMSE"]
            status = f"RMSE = {rmse_value:.4f}"

        log_entry = {
            "base_date": base_date.strftime("%Y-%m-%d"),
            "end_date": forecast_dates[-1].strftime("%Y-%m-%d"),
            "rmse": rmse_value,
            "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        all_logs.append(log_entry)

        print(
            f"Base date: {log_entry['base_date']} → End date: {log_entry['end_date']} "
            f"| {status} | Logged at: {log_entry['logged_at']}"
        )

    # --- Lưu log an toàn ---
    joblib.dump(all_logs, LOG_PATH)


# ---  Task tự động hàng ngày ---
def daily_update():
    print(f"Bắt đầu cập nhật dữ liệu lúc {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Lấy dữ liệu mới
    new_data = fetch_latest_weather_data()

    if new_data.empty:
        print(" Không có dữ liệu mới để cập nhật.")
        return

    # 2. Cập nhật file 3 năm
    update_three_year_data(new_data)

    # 3. Chuẩn bị dữ liệu & dự báo
    processed = prepare_data(new_data)
    # print("Processed_X shape:", getattr(processed, "shape", None))
    # print("Processed_X columns:", getattr(processed, "columns", None))
    # print("Processed_X empty?", getattr(processed, "empty", None))

    y_pred = predict_tomorrow(processed)

    # 4. Ghi log kết quả dự đoán hôm nay
    save_prediction_log(y_pred)
    log_rmse_daily(r'data\realtime_predictions.csv', r'data\Current_Raw_3weeks.csv')
    print(f" Dự báo hoàn tất, {len(y_pred)} giá trị được dự đoán.")
    print(f'{y_pred}')
    print("Cập nhật & dự báo hoàn tất.\n")

def save_prediction_log(y_pred, output_dir="data"):
    """Lưu dự đoán từng dòng, mỗi dòng là 1 bộ giá trị dự đoán.
    Đảm bảo hàng cuối cùng là ngày hôm nay, các hàng trước là các ngày trước đó."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "realtime_predictions.csv")

    today = datetime.now()
    weekday = today.strftime("%A")
    date_str = today.strftime("%Y-%m-%d")

    y_pred = y_pred.tolist() if hasattr(y_pred, "tolist") else y_pred
    n_days = len(y_pred)

    # Tạo danh sách ngày lùi về quá khứ theo số dòng dự đoán
    dates = [(today - timedelta(days=n_days - 1 - i)) for i in range(n_days)]
    weekdays = [d.strftime("%A") for d in dates]
    date_strings = [d.strftime("%Y-%m-%d") for d in dates]

    df_pred = pd.DataFrame(y_pred, columns=[f"pred_day_{i+1}" for i in range(len(y_pred[0]))])
    df_pred.insert(0, "date", date_strings)
    df_pred.insert(0, "weekday", weekdays)

    # Đọc file cũ (nếu có)
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        old_df = pd.read_csv(file_path)
        # Loại bỏ các ngày trùng lặp
        old_df = old_df[~old_df["date"].isin(df_pred["date"])]
        # Ghép và sắp xếp lại theo ngày
        df_pred = pd.concat([old_df, df_pred], ignore_index=True)
        df_pred = df_pred.sort_values(by="date").reset_index(drop=True)

    df_pred.to_csv(file_path, index=False)
    print(f"Lưu dự đoán dạng phẳng vào {file_path}")


# # --- Lên lịch chạy lúc 00:00 mỗi ngày ---
# schedule.every().day.at("12:00").do(daily_update)

if __name__ == "__main__":
    # Chạy thử ngay lập tức khi khởi động, bao giờ ổn thì xoá daily_update đi để nó tự chạy theo lịch thôi
    daily_update()

    # # print("Service đang chạy... sẽ tự động cập nhật mỗi ngày lúc 12:00.")
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60)


