import os
import joblib
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from datetime import datetime, timedelta
from src.model_training import retrain_pipeline

LOG_PATH = r"logs/daily_rmse.txt" # ouput của hàm daily_log_ trong phần forecasting
RETRAIN_LOG_PATH = r"logs/retrain_log.pkl"
DATA_PATH = r"data\latest_3_year.csv"

# --- Ngưỡng cho monitoring ---
RMSE_THRESHOLD = 2.5       # Ngưỡng cảnh báo drift về hiệu năng
RETRAIN_INTERVAL_DAYS = 90 # Tự động retrain sau 90 ngày


def check_rmse_drift(log_path=LOG_PATH, threshold=RMSE_THRESHOLD):
    """Kiểm tra xem RMSE trung bình 5 ngày gần nhất có vượt ngưỡng không"""
    if not os.path.exists(log_path):
        print("Không tìm thấy log metrics, bỏ qua kiểm tra drift.")
        return False

    log_data = joblib.load(log_path)
    if isinstance(log_data, list):
        df = pd.DataFrame(log_data)
        if "rmse" in df.columns:
            df = df.rename(columns={"rmse": "RMSE"}) # Đổi tên cột để tương thích
    elif isinstance(log_data, dict):
        df = pd.DataFrame(log_data)
    else:
        df = log_data # Giả sử đã là DataFrame

    # Tính RMSE trung bình 5 ngày gần nhất
    if "RMSE" in df.columns:
        # Lọc bỏ các giá trị None/NaN trước khi tính toán
        recent_rmse_series = df["RMSE"].dropna().tail(5)
        if recent_rmse_series.empty:
            print("Không có đủ dữ liệu RMSE hợp lệ trong 5 ngày gần nhất.")
            return False
            
        recent_rmse = recent_rmse_series.mean()
        print(f"RMSE 5 ngày gần nhất: {recent_rmse:.3f}")
        if recent_rmse > threshold:
            print(f"RMSE vượt ngưỡng {threshold}! => cần retrain")
            return True
    else:
        print("Log không có cột RMSE, bỏ qua kiểm tra drift.")
    return False


def check_retrain_interval(log_path=RETRAIN_LOG_PATH, interval_days=RETRAIN_INTERVAL_DAYS):
    """Kiểm tra xem đã quá 90 ngày kể từ lần retrain gần nhất chưa"""
    if not os.path.exists(log_path):
        print("Chưa có lịch sử retrain nào.")
        return False

    records = joblib.load(log_path)
    if not records:
        return True

    last_date_str = records[-1]["timestamp"]
    last_date = datetime.strptime(last_date_str, "%Y-%m-%d %H:%M:%S")
    days_since = (datetime.now() - last_date).days

    print(f"Lần retrain gần nhất: {last_date_str} ({days_since} ngày trước)")
    if days_since > interval_days:
        print(f"Đã quá {interval_days} ngày => retrain")
        return True
    return False


def monitor_and_retrain():
    """Giám sát và tự động retrain khi cần"""
    need_retrain = False

    # 1. Kiểm tra drift
    if check_rmse_drift():
        need_retrain = True

    # 2. Kiểm tra thời gian từ lần retrain gần nhất
    if check_retrain_interval():
        need_retrain = True

    # 3. Nếu cần retrain thì gọi pipeline
    if need_retrain: # True
        print("Tiến hành retrain model...")
        metrics = retrain_pipeline(DATA_PATH) # gọi hàm retrain trong model_training
        print("Retrain hoàn tất. Metrics:", metrics)
    else:
        print("Hệ thống ổn định, không cần retrain.")


if __name__ == "__main__":
    monitor_and_retrain()

# hàm check interval chạy ổn
# hàm check rmse, chạy thử với file daily_rmse có ổn ko
# xem có cần tự động 5 ngày monitorin, ... / hoặc là sau khi gọi bên forcasting => gọi hàm  monitor_and_retrain trong monitoring