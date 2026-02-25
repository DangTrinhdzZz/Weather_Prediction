"""
Model evaluation
- RMSE, R2, MAE
"""

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
import numpy as np
import pandas as pd

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    y_true_safe = np.where(y_true == 0, np.finfo(float).eps, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

def evaluate_multi_output(y_true, y_pred):
    """
    Đánh giá mô hình đa đầu ra bằng cách tính trung bình các chỉ số
    và báo cáo chi tiết cho từng đầu ra (từng ngày dự báo).

    Args:
        y_true (pd.DataFrame or np.array): Giá trị thực tế (n_samples, n_outputs).
        y_pred (pd.DataFrame or np.array): Giá trị dự báo (n_samples, n_outputs).

    Returns:
        dict: Một dictionary chứa các chỉ số trung bình và chi tiết cho từng ngày.
    """
    # Đảm bảo y_true và y_pred là numpy array để xử lý nhất quán
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    n_outputs = y_true.shape[1]
    results = {
        "average": {},  # Chỉ số trung bình
        "per_day": {}   # Chỉ số từng ngày
    }

    # --- Phần tính toán chi tiết ---
    rmse_scores, mae_scores, r2_scores, mape_scores = [], [], [], []
    for i in range(n_outputs):
        day = i + 1
        true_day_i = y_true[:, i]
        pred_day_i = y_pred[:, i]

        # Tính các chỉ số
        rmse = np.sqrt(mean_squared_error(true_day_i, pred_day_i))
        mae = mean_absolute_error(true_day_i, pred_day_i)
        r2 = r2_score(true_day_i, pred_day_i)
        #mape = mean_absolute_percentage_error(true_day_i, pred_day_i) * 100

        # Lưu vào danh sách
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        #mape_scores.append(mape)

        # Lưu kết quả chi tiết từng ngày
        results["per_day"][f"RMSE_day_{day}"] = rmse
        results["per_day"][f"MAE_day_{day}"] = mae
        results["per_day"][f"R2_day_{day}"] = r2
        #results["per_day"][f"MAPE_day_{day}"] = mape

    # --- Trung bình ---
    results["average"]["RMSE"] = np.mean(rmse_scores)
    results["average"]["MAE"] = np.mean(mae_scores)
    results["average"]["R2"] = np.mean(r2_scores)
    #results["average"]["MAPE"] = np.mean(mape_scores)

    return results