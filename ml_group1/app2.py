import os
import joblib
import pandas as pd
import numpy as np
import gradio as gr
from datetime import timedelta
import threading 

# --- Import project modules (assumes this file is at repo root) ---
try:
    from src.data_preprocessing import basic_preprocessing
    from src.new_feature_engineering_daily import feature_engineering
except Exception:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
    from src.data_preprocessing import basic_preprocessing
    from src.new_feature_engineering_daily import feature_engineering
try:
    from src.monitoring import monitor_and_retrain
except Exception:
    from src.monitoring import monitor_and_retrain
try:
    import main as scheduler_main          # main.py ở project root
except Exception:
    from src import main as scheduler_main # fallback nếu để trong src/

REALTIME_CSV = os.getenv("REALTIME_CSV", "data/realtime_predictions.csv")

MODEL_PATH = os.getenv("MODEL_PATH", "models/Current_model.pkl")
PIPE_1 = os.getenv("PIPE_1", "pipelines/preprocessing_pipeline.pkl")
PIPE_2 = os.getenv("PIPE_2", "pipelines/featureSelection_pipeline.pkl")
DEFAULT_CSV = os.getenv("DEFAULT_CSV", "data/latest_3_year.csv")

_model, _pipe1, _pipe2 = None, None, None

def load_artifacts():
    global _model, _pipe1, _pipe2
    if _pipe1 is None:
        _pipe1 = joblib.load(PIPE_1)
    if _pipe2 is None:
        _pipe2 = joblib.load(PIPE_2)
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model, _pipe1, _pipe2

def _postprocess_prediction(dates, y_pred):
    out = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "horizon_day": np.arange(1, len(y_pred) + 1),
        "pred_temp (°C)": np.round(np.asarray(y_pred, dtype=float), 2),
    })
    return out

def _make_plot(df_out: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=120)
    ax.plot(df_out["date"], df_out[df_out.columns[-1]], marker="o", color="#0ea5e9", linewidth=2)
    ax.fill_between(df_out["date"], df_out[df_out.columns[-1]], color="#0ea5e922")
    ax.set_title("🌡️ 5-day Temperature Forecast", fontsize=12, fontweight="bold", color="#0369a1")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def show_latest_realtime():
    if not os.path.exists(REALTIME_CSV):
        return None, gr.update(value="⚠️ Chưa thấy data/realtime_predictions.csv"), None
    try:
        df = pd.read_csv(REALTIME_CSV)  # file của bạn có header: weekday, date, pred_day_1..5
    except Exception:
        df = pd.read_csv(REALTIME_CSV, header=None)

    if df.empty:
        return None, gr.update(value="⚠️ File realtime_predictions.csv rỗng"), None

    last = df.iloc[-1]
    # nếu có header chuẩn:
    base_date = pd.to_datetime(last.get("date", pd.to_datetime(last[1], errors="coerce"))).normalize()
    # gom các cột dự báo pred_day_*
    try:
        preds = [float(last[c]) for c in df.columns if str(c).startswith("pred_day_")]
    except Exception:
        # fallback: lấy các giá trị số còn lại
        vals = last.values.tolist()
        preds = []
        for v in vals:
            try:
                preds.append(float(v))
            except Exception:
                pass

    if not preds:
        return None, gr.update(value="⚠️ Không tách được giá trị dự báo từ realtime_predictions.csv"), None

    horizon_dates = [(base_date + pd.Timedelta(days=i+1)).date() for i in range(len(preds))]
    df_out = _postprocess_prediction(horizon_dates, preds)
    fig = _make_plot(df_out)
    # summary an toàn: dùng vị trí cột
    summary = "\n".join([f"Day +{r.horizon_day} ({r.date.date()}): {r._3} °C" for r in df_out.itertuples()])
    return df_out, summary, fig

    
def show_latest_realtime():
    # Đọc dòng mới nhất trong data/realtime_predictions.csv và hiển thị
    if not os.path.exists(REALTIME_CSV):
        return None, gr.update(value="⚠️ Chưa thấy data/realtime_predictions.csv"), None

    # File của bạn thường có header: weekday, date, pred_day_1..pred_day_5
    try:
        df = pd.read_csv(REALTIME_CSV)
    except Exception:
        df = pd.read_csv(REALTIME_CSV, header=None)

    if df.empty:
        return None, gr.update(value="⚠️ File realtime_predictions.csv rỗng"), None

    last = df.iloc[-1]
    # base_date
    base_date = pd.to_datetime(last.get("date", pd.to_datetime(last[1], errors="coerce"))).normalize()
    # lấy các cột pred_day_*
    try:
        preds = [float(last[c]) for c in df.columns if str(c).startswith("pred_day_")]
    except Exception:
        vals = last.values.tolist()
        preds = []
        for v in vals:
            try:
                preds.append(float(v))
            except Exception:
                pass

    if not preds:
        return None, gr.update(value="⚠️ Không tách được giá trị dự báo từ realtime_predictions.csv"), None

    horizon_dates = [(base_date + pd.Timedelta(days=i+1)).date() for i in range(len(preds))]
    df_out = _postprocess_prediction(horizon_dates, preds)
    fig = _make_plot(df_out)
    summary = "\n".join([f"Day +{r.horizon_day} ({r.date.date()}): {r._3} °C" for r in df_out.itertuples()])
    return df_out, summary, fig

def run_daily_background_then_show_now():
    # chạy daily_job ở nền
    job = getattr(scheduler_main, "daily_job", None) or getattr(scheduler_main, "do_daily", None)
    if job is not None:
        threading.Thread(target=job, daemon=True).start()

    # đọc realtime → df_out
    df_out, summary, _ = show_latest_realtime()
    if df_out is None:
        return 0, None, gr.update(value=str(summary)), None  # state=0

    # render ngày đầu
    html_cards, idx = _build_cards_html(df_out, 0)
    html_map = _build_map_html()
    return idx, html_cards, summary, html_map


# === Render helpers (card + map) ===
def _temp_icon_and_style(t):
    """Chọn icon & màu theo ngưỡng °C."""
    # có thể tinh chỉnh ngưỡng cho HN
    if t < 12:      # rất lạnh
        return "🥶", "#60a5fa"
    if t < 18:      # lạnh - mát
        return "🧊", "#38bdf8"
    if t < 24:      # mát
        return "⛅", "#0ea5e9"
    if t < 30:      # ấm
        return "🌤️", "#22c55e"
    if t < 35:      # nóng
        return "🌞", "#f59e0b"
    return "🔥", "#ef4444"  # rất nóng

def _one_day_card_html(date_str, temp_c):
    icon, color = _temp_icon_and_style(temp_c)
    return f"""
    <div class="wx-card">
      <div class="wx-date">{date_str}</div>
      <div class="wx-main">
        <div class="wx-icon" style="background: radial-gradient(220px 90px at 20% 10%, {color}22, transparent);">
          {icon}
        </div>
        <div class="wx-temp">
          <div class="wx-t">{temp_c:.1f}°C</div>
          <div class="wx-desc">Hanoi • 5-day forecast</div>
        </div>
      </div>
    </div>
    """

def _build_cards_html(df_out, idx):
    """Tạo block cho ngày đang chọn + thanh ngày nhỏ."""
    n = len(df_out)
    idx = int(idx) % n
    row = df_out.iloc[idx]
    main = _one_day_card_html(row["date"].date().isoformat(), float(row[df_out.columns[-1]]))

    # thanh chọn ngày nhỏ (pill)
    pills = []
    for i, r in df_out.iterrows():
        active = "active" if i == idx else ""
        pills.append(f"""
        <div class="pill {active}">
          <div>{pd.to_datetime(r['date']).strftime('%d/%m')}</div>
          <div>{float(r[df_out.columns[-1]]):.0f}°</div>
        </div>
        """)
    strip = "<div class='pill-strip'>" + "".join(pills) + "</div>"
    return f"<div class='cards-wrap'>{main}{strip}</div>", idx

def _build_map_html(lat=21.0278, lon=105.8342, zoom=11):
    # bbox đơn giản quanh toạ độ để hiển thị vùng Hà Nội
    bbox = f"{lon-0.15}%2C{lat-0.12}%2C{lon+0.15}%2C{lat+0.12}"
    return f'''
    <iframe
      width="100%" height="340"
      style="border:0;border-radius:16px"
      src="https://www.openstreetmap.org/export/embed.html?bbox={bbox}&layer=mapnik&marker={lat}%2C{lon}">
    </iframe>
    <div style="margin-top:6px;font-size:12px;color:#475569">
      Map © <a href="https://www.openstreetmap.org">OpenStreetMap</a> contributors
    </div>
    '''


def go_prev(idx, df_out):
    if df_out is None:
        return idx, None
    idx = (int(idx) - 1) % len(df_out)
    html, idx = _build_cards_html(df_out, idx)
    return idx, html

def go_next(idx, df_out):
    if df_out is None:
        return idx, None
    idx = (int(idx) + 1) % len(df_out)
    html, idx = _build_cards_html(df_out, idx)
    return idx, html

with gr.Blocks(
    title="Hanoi Temperature Forecast – 5 days",
    theme=gr.themes.Soft(primary_hue="sky", neutral_hue="slate"),
    css="""
    body{
      background:
        radial-gradient(1200px 480px at 12% -10%, #dbeafe88, transparent),
        radial-gradient(1200px 520px at 88% -12%, #c7d2fe88, transparent),
        linear-gradient(180deg, #f1f5f9 0%, #e9efff 45%, #e6edff 100%);
    }
    .gradio-container{max-width:1180px}
    .hero{padding:18px;border-radius:18px;background:linear-gradient(135deg,#60a5fa22,#7dd3fc22);
          border:1px solid #93c5fd55;box-shadow:0 6px 18px rgba(2,132,199,.08)}
    .hero h1{margin:0;font-size:28px;font-weight:800;color:#0c4a6e;letter-spacing:.3px}
    .hero p{margin:6px 0 0;color:#0ea5e9}

    .cards-wrap{background:#ffffffdd;border:1px solid #e2e8f0;border-radius:18px;padding:14px}
    .wx-card{display:flex;flex-direction:column;gap:6px}
    .wx-date{font-weight:700;color:#0369a1}
    .wx-main{display:flex;align-items:center;gap:14px}
    .wx-icon{font-size:54px;width:92px;height:92px;display:flex;align-items:center;justify-content:center;border-radius:16px}
    .wx-temp .wx-t{font-size:44px;font-weight:800;color:#0c4a6e;line-height:1}
    .wx-temp .wx-desc{color:#475569}
    .pill-strip{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
    .pill{border-radius:12px;border:1px solid #e5e7eb;padding:6px 10px;background:#f8fafc;color:#0f172a}
    .pill.active{background:#dbeafe;border-color:#93c5fd}
    .nav-btn{border-radius:12px !important;background:#0284c7 !important;color:#fff !important}
    .card{background:#ffffffd9;border:1px solid #e2e8f0;border-radius:16px;padding:10px}
    """
) as demo:

    gr.HTML("""
      <div class="hero">
        <h1>❄️ Hanoi Temperature Forecast — 5 days</h1>
        <p>App hiển thị dự báo mới nhất (realtime) và tự cập nhật + retrain ở nền.</p>
      </div>
    """)

    # state và dữ liệu
    idx_state = gr.State(0)             # index đang xem
    df_state = gr.State()               # giữ df_out để chuyển ngày

    with gr.Row():
        with gr.Column(scale=3):
            cards_html = gr.HTML(elem_classes=["card"], label="Dự báo")
            with gr.Row():
                prev_btn = gr.Button("◀", elem_classes=["nav-btn"])
                next_btn = gr.Button("▶", elem_classes=["nav-btn"])
        with gr.Column(scale=1):
            map_html = gr.HTML(elem_classes=["card"])
            summary_box = gr.Textbox(label="Tóm tắt", lines=10, elem_classes=["card"])

    # Autoload: chạy job nền + render ngày đầu
    def _load():
        i, cards, summary, map_ = run_daily_background_then_show_now()
        # đồng thời, lấy df_out để điều hướng trái/phải
        df_out, _, _ = show_latest_realtime()
        return i, df_out, cards, summary, map_

    demo.load(fn=_load, inputs=None, outputs=[idx_state, df_state, cards_html, summary_box, map_html])

    # Nút trái/phải
    prev_btn.click(go_prev, inputs=[idx_state, df_state], outputs=[idx_state, cards_html])
    next_btn.click(go_next, inputs=[idx_state, df_state], outputs=[idx_state, cards_html])


if __name__ == "__main__":
    demo.launch(share=True)
