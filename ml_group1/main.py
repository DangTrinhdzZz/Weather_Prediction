print(">>> File main.py bắt đầu chạy...") 
from pathlib import Path
import gradio as gr
import time
import schedule
from src import forecasting as fc
from src import monitoring as mo
from datetime import datetime


def daily_job():
    print(f"\n=== BẮT ĐẦU DỰ BÁO LÚC {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    try:
        fc.daily_update()
        print("Dự báo hoàn tất.")
        mo.monitor_and_retrain()
        print("Monitoring & retrain (nếu cần) hoàn tất.")
    except Exception as e:
        print(f"Lỗi trong quá trình chạy daily job: {e}")
    print("=== HOÀN TẤT CHU KỲ ===\n")


# --- Lên lịch chạy mỗi ngày lúc 12:00 ---
schedule.every().day.at("12:00").do(daily_job)

if __name__ == "__main__":
    print(" Service khởi động. Sẽ tự động chạy forecast + monitoring mỗi ngày lúc 12:00.")
    # daily_job()

    while True:
        schedule.run_pending()
        time.sleep(60)
