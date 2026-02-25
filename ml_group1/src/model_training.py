"""
Model Training for time series
- Uses time-based train/test split
- Trains a RandomForestRegressor
- Handles categorical features via one-hot encoding
- Saves model
"""
import joblib
import pandas as pd
import clearml
import random
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optuna
import numpy as np
import shutil
import logging

from pathlib import Path
from clearml import Logger, Task
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from lightgbm import LGBMRegressor

from src.data_preprocessing import load_data, basic_preprocessing
from src.pipeline import build_preprocessing_pipeline
from src.new_feature_engineering_daily import feature_engineering
from src.model_evaluation import evaluate_multi_output, evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = r"data\latest_3_year.csv"
OLD_MODEL_PATH = r"models\Current_model.pkl"
NEW_MODEL_PATH = r"models\Update_model.pkl"
PIPE_1 = r"pipelines\preprocessing_pipeline.pkl"
#PIPE_2 = r"pipelines\featureSelection_pipeline.pkl"

def load_new_data(data_path: str):
   """Load Data 3 năm gần nhất để retrain model"""
   df = load_data(data_path)
   return df

def preprocess_data(df):
   """Gọi pipeline preprocessing để xử lý data"""

   # 1. basic preprocessing for all data set
   df = basic_preprocessing(df=df)

   # 2. Chia train, test
   train_size = 0.8
   n = len(df)
   train_df = df.iloc[:int(train_size * n)]
   test_df = df.iloc[int(train_size * n):]

   # 3. Pipeline 1: preprocessing
   pipeline1 = build_preprocessing_pipeline()
   pipeline1.fit(train_df)

   train_processed = pipeline1.transform(train_df)
   test_processed = pipeline1.transform(test_df)
   joblib.dump(pipeline1, "pipelines/preprocessing_pipeline.pkl")

   # 4. Feature engineering
   train_feat, target_col = feature_engineering(train_processed)
   test_feat, _ = feature_engineering(test_processed)

   # 5. Chia X,y
   X_train = train_feat.drop(columns= target_col)
   y_train = train_feat[target_col]

   X_test = test_feat.drop(columns= target_col)
   y_test = test_feat[target_col]

   # # 6. Pipeline 2: GB selection
   # pipeline2 = build_GB_featture_engineering_pipeline(top_k= 35)
   # X_train_sel = pipeline2.fit_transform(X_train, y_train['temp_next_1'])
   # X_test_sel = pipeline2.transform(X_test)
   # joblib.dump(pipeline2, "pipelines/featureSelection_pipeline.pkl")

   # 7. lưu lại processed data
   save_dir = "data"
   os.makedirs(save_dir, exist_ok=True)

   X_train.to_csv(f"{save_dir}/New_X_train_sel.csv", index=False)
   y_train.to_csv(f"{save_dir}/New_y_train.csv", index=False)
   X_test.to_csv(f"{save_dir}/New_X_test_sel.csv", index=False)
   y_test.to_csv(f"{save_dir}/New_y_test.csv", index=False)
   
   return X_train, y_train, X_test, y_test

def train_model(X_train, y_train, random_state=42, n_trials=5):
   """Training LightGBM MultiOutputRegressor với Optuna tuning."""
   # === 1. Khởi tạo ClearML Task cho tuning session ===
   task = Task.init(
        project_name="Hanoi Temperature Forecast",
        task_name=f"Tuning_{datetime.now():%Y%m%d_%H%M}",
        tags=["optuna", "tuning", "pruning_enabled"] # Thêm tag để dễ nhận biết
   )
   logger = Logger.current_logger()

   def objective(trial):
      boosting_type = trial.suggest_categorical('boosting_type', ['gbdt','dart'])
      params = {
            'boosting_type': boosting_type,
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.3),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_bin': trial.suggest_int('max_bin', 64, 512),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'objective': 'regression',
            'metric': 'rmse',
            'random_state': random_state,
            'n_jobs': -1,
            'verbose': -1
        }

      if boosting_type == 'dart':
            params['drop_rate'] = trial.suggest_float('drop_rate', 0.05, 0.5)
            params['skip_drop'] = trial.suggest_float('skip_drop', 0.3, 0.7)

      cv = TimeSeriesSplit(n_splits=5)
      rmse_scores = []

      for step, (train_idx, val_idx) in enumerate(cv.split(X_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            base_model = LGBMRegressor(**params)
            model = MultiOutputRegressor(estimator=base_model, n_jobs=-1)
            model.fit(X_tr, y_tr)

            y_pred_val = model.predict(X_val)
            metrics = evaluate_multi_output(y_val, y_pred_val)
            rmse_score_fold = metrics["average"]["RMSE"] # Lấy RMSE của fold hiện tại
            rmse_scores.append(rmse_score_fold)
            
            # 1. Báo cáo kết quả trung gian của fold này cho trial
            trial.report(rmse_score_fold, step)

            # 2. Kiểm tra xem trial này có nên bị cắt tỉa không
            if trial.should_prune():
                # Ném ra ngoại lệ để Optuna dừng trial này và chuyển sang trial tiếp theo
                raise optuna.exceptions.TrialPruned()

      # === Log RMSE của trial lên ClearML ===
      mean_rmse = np.mean(rmse_scores)
      logger.report_scalar(
            title="Optuna Trials",
            series="RMSE",
            value=mean_rmse,
            iteration=trial.number
        )
      return mean_rmse

    # === 2. Chạy tuning Optuna ===
   sampler = optuna.samplers.TPESampler(seed=42)
   pruner = optuna.pruners.MedianPruner(n_warmup_steps=2) # Thêm Pruner
   study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
   
   study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
   print('Finished tuning')

   # === 3. Train lại mô hình tốt nhất trên toàn bộ tập train ===
   # Xử lý trường hợp study không có trial thành công (do pruning)
   pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
   complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

   if len(complete_trials) == 0:
        # Nếu tất cả các trial đều bị pruned, không có trial tốt nhất.
        # Chúng ta cần một phương án dự phòng.
        logging.error("Tất cả các trials đã bị pruned! Không tìm thấy tham số tốt nhất.")
        logging.warning("Sẽ không có mô hình mới nào được tạo.")
        # Trả về None để báo hiệu cho quy trình bên ngoài rằng training đã thất bại
        return None, None, task

   best_trial = study.best_trial
   logging.info(f"Best trial found: Trial #{best_trial.number} with RMSE = {best_trial.value:.4f}")

   best_params = best_trial.params
   best_model_base = LGBMRegressor(**best_params) # Sửa tên biến để rõ ràng hơn
   model = MultiOutputRegressor(estimator=best_model_base, n_jobs=-1)
   model.fit(X_train, y_train)

   # === 4. Lưu model
   joblib.dump(model, NEW_MODEL_PATH)
   print('Đã lưu model mới')

   # Trả về model, best_params, và task để log tiếp
   return model, best_params, task


def evaluate_model(model, X_test_sel, y_test):
   """Đánh giá performance của model"""
   y_preds = model.predict(X_test_sel)

   # --- Kiểm tra xem là multi-output hay single-output ---
   if len(y_preds.shape) > 1 and y_preds.shape[1] > 1:
        metrics = evaluate_multi_output(y_test, y_preds)
   else:
        metrics = evaluate(y_test, y_preds)

   return metrics


def save_artifacts(model, best_param, metrics, task, output_dir="logs"):
   """Lưu model & pipeline & log metrics"""
   os.makedirs(output_dir, exist_ok=True)

   # Lưu metrics ra file
   with open(f"{output_dir}/metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

   # Log final metrics
   # artifact
   task.upload_artifact("metrics", artifact_object= r'logs\metrics.txt')

   # scalar
   for k, v in metrics.items():
      if isinstance(v, dict):  # ví dụ: {"RMSE": 1.2, "R2": 0.9}
         for sub_k, sub_v in v.items():
               if isinstance(sub_v, (int, float)):
                  Logger.current_logger().report_scalar(
                     title=f"metrics_{k}", series=sub_k, value=sub_v, iteration=0
                  )
      elif isinstance(v, (int, float)):
         Logger.current_logger().report_scalar(
               title="metrics", series=k, value=v, iteration=0
         )


   # Log artifacts
   task.upload_artifact("model", artifact_object=NEW_MODEL_PATH)
   task.upload_artifact("best_parameters", artifact_object=best_param)

   # Lưu retrain history
   record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "best_params": best_param
   }
   retrain_log_path = os.path.join(output_dir, "retrain_log.pkl")
   if os.path.exists(retrain_log_path):
        old_records = joblib.load(retrain_log_path)
   else:
        old_records = []
   old_records.append(record)
   joblib.dump(old_records, retrain_log_path)

   # Đóng task
   task.close()
   
   return

def compare_models(new_model_path, old_model_path, X_test, y_test, threshold=0.99):
   """
   So sánh mô hình mới và cũ
   Sử dụng logging để ghi lại tiến trình.
   """
   if not os.path.exists(old_model_path):
      # logging.info("Không tìm thấy mô hình cũ. Tự động chấp nhận mô hình mới.")//
      new_model = joblib.load(new_model_path)
      metrics = evaluate_model(new_model, X_test, y_test)
      return True, metrics, None # is_better, new_metrics, old_metrics

   try:
      logging.info("Bắt đầu so sánh mô hình mới và mô hình cũ...")
      new_model = joblib.load(new_model_path)
      old_model = joblib.load(old_model_path)

      # logging.info("Đánh giá mô hình mới trên tập test...")
      new_metrics = evaluate_model(new_model, X_test, y_test)
      new_rmse = new_metrics["average"]["RMSE"]
      logging.info(f"-> RMSE mô hình mới: {new_rmse:.4f}")

      # logging.info("Đánh giá mô hình cũ trên tập test...")
      old_metrics = evaluate_model(old_model, X_test, y_test)
      old_rmse = old_metrics["average"]["RMSE"]
      logging.info(f"-> RMSE mô hình cũ: {old_rmse:.4f}")

      if new_rmse < old_rmse * threshold:
         improvement = (1 - new_rmse / old_rmse) * 100
         logging.info(f"QUYẾT ĐỊNH: Chấp nhận mô hình mới (Tốt hơn {improvement:.2f}%)")
         return True, new_metrics, old_metrics
      else:
         logging.info("QUYẾT ĐỊNH: Giữ lại mô hình cũ (Không đủ cải thiện)")
         return False, new_metrics, old_metrics

   except Exception as e:
      logging.error(f"Lỗi xảy ra trong quá trình so sánh mô hình: {e}")
      logging.warning("QUYẾT ĐỊNH: Giữ lại mô hình cũ để đảm bảo an toàn.")
      return False, None, None

def retrain_pipeline(data_path):
   """
   Full pipeline: load data -> prepare data -> train, predict -> eval -> save
   """
   # 1. Load data
   logging.info("Bắt đầu quy trình huấn luyện lại...")
   df = load_new_data(data_path)

   # 2. Prepare data
   # logging.info("Chuẩn bị và tiền xử lý dữ liệu...")
   X_train_sel, y_train, X_test_sel, y_test = preprocess_data(df)

   # 3. Train và Tune model
   # logging.info("Bắt đầu quá trình tuning và huấn luyện mô hình mới...")
   new_model, best_param, task = train_model(X_train_sel, y_train)
   # logging.info("Huấn luyện mô hình mới hoàn tất.")

   # 4. So sánh mô hình
   is_new_model_better, new_metrics, old_metrics = compare_models(
      new_model_path=NEW_MODEL_PATH,
      old_model_path=OLD_MODEL_PATH,
      X_test=X_test_sel,
      y_test=y_test
   )

   # 5. Quyết định triển khai và lưu artifacts
   deployment_decision = "kept_old_model"
   final_metrics = old_metrics

   if is_new_model_better:
      logging.info(f"Triển khai mô hình mới: Copy '{NEW_MODEL_PATH}' sang '{OLD_MODEL_PATH}'")
      shutil.copy(NEW_MODEL_PATH, OLD_MODEL_PATH) # Cập nhật mô hình hiện tại
      deployment_decision = "deployed_new_model"
      final_metrics = new_metrics

   # Tải lại model cuối cùng để đảm bảo nhất quán
   final_model = joblib.load(OLD_MODEL_PATH)

   # Log quyết định và lưu artifacts
   task.set_parameters({"Deployment Decision": deployment_decision})
   save_artifacts(final_model, best_param, final_metrics, task=task)
   # logging.info("Lưu artifacts và log kết quả lên ClearML hoàn tất.")

   # 6. Trả về kết quả để hàm gọi có thể hiển thị
   return deployment_decision, final_metrics, new_metrics, old_metrics

def main():
   print("===================================================")
   print("    BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN LẠI MÔ HÌNH       ")
   print("===================================================")

   # Chạy pipeline chính
   results = retrain_pipeline(data_path=DATA_PATH)
   
   # Kiểm tra xem pipeline có chạy thành công không
   if results is None:
      print("\n[LỖI] Quy trình huấn luyện lại đã thất bại. Vui lòng kiểm tra logs.")
      return

   deployment_decision, final_metrics, new_metrics, old_metrics = results

   print("\n===================================================")
   print("              KẾT QUẢ HUẤN LUYỆN LẠI              ")
   print("===================================================")

   if deployment_decision == "deployed_new_model":
      print("\n[QUYẾT ĐỊNH]: Triển khai MÔ HÌNH MỚI thành công!")
      if old_metrics:
            improvement = (1 - new_metrics["average"]["RMSE"] / old_metrics["average"]["RMSE"]) * 100
            print(f"-> Cải thiện so với mô hình cũ: {improvement:.2f}%")
      else:
            print("-> Đây là mô hình đầu tiên được triển khai.")
   else:
      print("\n[QUYẾT ĐỊNH]: Giữ lại MÔ HÌNH CŨ.")
      if new_metrics:
         print("-> Mô hình mới không đạt đủ ngưỡng cải thiện.")

   print("\n--- Metrics của mô hình được triển khai ---")
   if final_metrics:
      print(f"  RMSE (trung bình): {final_metrics['average']['RMSE']:.4f}")
      print(f"  R2 (trung bình):   {final_metrics['average']['R2']:.4f}")
      print(f"  MAE (trung bình): {final_metrics['average']['MAE']:.2f}%")
   else:
      print("  Không có metrics để hiển thị.")

   print("\n===================================================")
   print("            QUY TRÌNH ĐÃ HOÀN TẤT                 ")
   print("===================================================")


if __name__ == "__main__":
   main()