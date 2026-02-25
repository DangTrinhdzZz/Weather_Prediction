import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import base64
import requests
import textwrap

from dotenv import load_dotenv
load_dotenv()

try:
    from src.forecasting import daily_update
except ImportError:
    # st.error("Lỗi import: Không tìm thấy các hàm từ thư mục 'src'. Vui lòng kiểm tra lại cấu trúc thư mục của bạn.")
    def daily_update():
        st.warning("Hàm daily_update() không được tìm thấy. Chức năng cập nhật sẽ không hoạt động.")
        return None

# --- CÁC ĐƯỜNG DẪN TỚI FILE ---
PATH_PREDICTIONS = r'data/realtime_predictions.csv'
PATH_RAW_3WEEKS = r'data/Current_Raw_3weeks.csv'
PATH_3_YEAR_DATA = r'data/latest_3_year.csv'
PATH_RMSE_LOG = r'logs/daily_rmse.txt'
PATH_RETRAIN_LOG = r'logs/retrain_log.pkl'
PATH_WEATHER_ICON = r'assets/sun.png'


# --- HÀM HỖ TRỢ VỚI CACHING ---
@st.cache_data(ttl=3600)
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data(ttl=3600)
def load_joblib(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None

def get_img_as_base64(file):
    with open(file, "rb") as f: data = f.read()
    return base64.b64encode(data).decode()

def load_keys_from_env():
    """Tải danh sách Visual Crossing API keys từ file .env."""
    load_dotenv()
    keys_string = os.getenv("VISUAL_CROSSING_API_KEYS")
    if keys_string:
        return [key.strip() for key in keys_string.split(',')]
    else:
        # Hiển thị lỗi một lần duy nhất khi ứng dụng khởi động nếu không tìm thấy key
        st.error("Lỗi cấu hình: Biến 'VISUAL_CROSSING_API_KEYS' không được tìm thấy trong file .env.")
        return []

@st.cache_data(ttl=900) # Cache kết quả trong 15 phút
def fetch_realtime_weather(location="Hanoi", api_keys=None):
    """
    Gọi API Visual Crossing để lấy dữ liệu thời tiết hiện tại.
    Tự động xoay vòng qua danh sách API keys nếu gặp lỗi.
    """
    if not api_keys:
        return None # Không có key nào để thử

    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    url = f"{base_url}/{location}/today"

    for api_key in api_keys:
        params = {
            "unitGroup": "metric",
            "include": "current",
            "key": api_key,
            "contentType": "json"
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            current_data = data.get("currentConditions")
            
            if not current_data:
                continue # Dữ liệu không hợp lệ, thử key tiếp theo

            # Trích xuất dữ liệu và trả về khi thành công
            return {
                "temperature": current_data.get("temp"),
                "feels_like": current_data.get("feelslike"),
                "chance_of_rain": current_data.get("precipprob"),
                "wind_speed": current_data.get("windspeed"),
                "uv_index": current_data.get("uvindex"),
                "humidity": current_data.get("humidity"),
                "conditions": current_data.get("conditions")
            }

        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code in [401, 429]:
                # Lỗi sai key hoặc hết hạn ngạch -> thử key tiếp theo
                continue
            else:
                st.error(f"Lỗi HTTP nghiêm trọng: {http_err}")
                return None # Dừng lại nếu là lỗi server
        except Exception:
            # Lỗi mạng hoặc lỗi không xác định khác -> thử key tiếp theo
            continue

    st.error("Tất cả các API key đều thất bại. Vui lòng kiểm tra lại.")
    return None

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Hanoi Temperature Forecast",
    page_icon="☀️",
    layout="wide"
)

# --- CSS TÙY CHỈNH CHO SIDEBAR THEO MẪU MỚI ---
st.markdown("""
<style>
    /* Sidebar container */
    [data-testid="stSidebar"][aria-expanded="true"] {
        background-color: #1F242D;
        width: 230px;
        min-width: 230px;
        max-width: 230px;
        border-right: none;
    }

    /* Giảm chiều cao của sidebarHeader để bớt trống */
    [data-testid="stSidebarHeader"] {
        padding-top: 1rem;
        padding-bottom: 0rem;
        min-height: 0px;
        height: 0px;
    }

    /* Vùng chứa nội dung bên trong sidebar */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.8rem;
        padding-bottom: 0.8rem;
        padding-left: 0;
        padding-right: 0;
    }
    
    /* Tiêu đề Menu */
    [data-testid="stSidebar"] h1 {
        color: #FFFFFF;
        font-size: 1.3rem;
        margin-top: 0;
        margin-bottom: 0.4rem;
        padding-left: 0.8rem;
    }
    
    /* CSS cho tất cả các nút trong sidebar */
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        border: none;
        padding: 10px 8px;
        text-align: left !important;
        font-size: 10px;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: none !important; 
        border-radius: 3px; 
        margin-left: 0px;
        margin-right: 0px;
    }

    /* Nút KHÔNG được chọn */
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background-color: transparent;
        color: #A0AEC0;
    }

    /* Nút KHÔNG được chọn khi di chuột qua */
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background-color: #2C313A;
        color: #FFFFFF;
    }

    /* Nút ĐƯỢC CHỌN */
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: transparent;
        color: #FFFFFF;
        font-weight: 600;
        border-left: 3px solid #007BFF; 
    }
            
    /* CSS cho realtime weather block */
    .main-info-block {
        background: #1F242D;
        padding: 1.2rem;
        border-radius: 24px;
        margin-bottom: 1.5rem;
    }
    
    .city-name {
        background-color: #007BFF;
        border-radius: 16px;
        font-size: 0.85rem !important; 
        font-weight: 500 !important;
        color: #FFFFFF !important;
        margin: 0 0 0.5rem 0 !important;
        padding: 3px 10px !important;
        text-align: center !important;
        display: inline-block !important;
    }

    .day-of-week {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        margin: 0 0 0.2rem 0 !important;
        padding: 0 !important;
    }

    .date-time {
        font-size: 0.9rem !important;
        color: rgba(255, 255, 255, 0.7) !important;
        margin: 0 0 0.6rem 0 !important;
        padding: 0 !important;
    }
            
    .big-temp {
        font-size: 3.5rem !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
    }

    /* CSS cho weather icon và condition */
    .weather-icon-wrapper {
        text-align: right;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        justify-content: center;
    }

    .weather-icon-wrapper img {
        width: 100px;
        height: 100px;
        max-width: 100%;
    }

    /* CSS cho weather details block */
    .weather-details-block {
        background: linear-gradient(135deg, #1F242D 20%, #11332B 80%);
        padding: 1.2rem;
        border-radius: 24px;
        margin-bottom: 1.5rem;
    }

    .detail-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.7rem;
    }

    .detail-item {
        background: rgba(255, 255, 255, 0.05);
        padding: 0.8rem 0.6rem;
        border-radius: 12px;
        text-align: center;
    }

    .detail-label {
        font-size: 0.75rem !important;
        color: rgba(255, 255, 255, 0.6) !important;
        margin: 0 0 0.3rem 0 !important;
        padding: 0 !important;
    }

    .detail-value {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* CSS cho weather condition text */
    .weather-condition {
        font-size: 0.95rem !important;
        color: rgba(255, 255, 255, 0.8) !important;
        text-align: right !important;
        margin-top: 0.4rem !important;
        font-weight: 500 !important;
    }

    /* Responsive adjustments */
    @media (max-width: 1200px) {
        .day-of-week {
            font-size: 1.5rem !important;
        }
        .big-temp {
            font-size: 3rem !important;
        }
        .weather-icon-wrapper img {
            width: 80px;
            height: 80px;
        }
    }

    /* CSS cho forecast block */
    .forecast-block {
        background: #1F242D;
        padding: 1.5rem;
        border-radius: 24px;
        margin-bottom: 1.5rem;
    }

    .forecast-title {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        margin: 0 0 1rem 0 !important;
        padding: 0 !important;
    }

    .forecast-cards {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.8rem;
        margin-bottom: 1.5rem;
    }

    .forecast-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem 0.5rem;
        border-radius: 16px;
        text-align: center;
        transition: transform 0.2s ease, background 0.2s ease;
    }

    .forecast-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.08);
    }

    .forecast-day {
        font-size: 0.85rem !important;
        color: rgba(255, 255, 255, 0.7) !important;
        margin: 0 0 0.3rem 0 !important;
        font-weight: 500 !important;
    }

    .forecast-date {
        font-size: 0.75rem !important;
        color: rgba(255, 255, 255, 0.5) !important;
        margin: 0 0 0.8rem 0 !important;
    }

    .forecast-temp {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        margin: 0 !important;
    }

    .forecast-chart-container {
        background: rgba(255, 255, 255, 0.03);
        padding: 1rem;
        border-radius: 16px;
    }

    @media (max-width: 1200px) {
        .forecast-cards {
            grid-template-columns: repeat(3, 1fr);
        }
    }

    @media (max-width: 768px) {
        .forecast-cards {
            grid-template-columns: repeat(2, 1fr);
        }
    }
</style>
""", unsafe_allow_html=True)

# --- SỬ DỤNG SIDEBAR VỚI LOGIC BUTTON ĐÃ CẢI TIẾN ---
with st.sidebar:
    st.title("Main Menu")
    
    PAGES = {
        "Forecasting": "☀️",
        "Historical Data Analysis": "📊",
        "Model Performance": "⚙️",
    }
    
    if 'page_selection' not in st.session_state:
        st.session_state.page_selection = "Forecasting"
    
    # Tạo các nút bấm bằng một vòng lặp để code gọn hơn
    for page_name, icon in PAGES.items():
        # Dùng type="primary" cho nút được chọn, "secondary" cho các nút còn lại
        # Đây là cách để CSS có thể phân biệt và định dạng chúng
        is_selected = (st.session_state.page_selection == page_name)
        button_type = "primary" if is_selected else "secondary"
        
        if st.button(f"{icon} {page_name}", type=button_type):
            st.session_state.page_selection = page_name
            st.rerun()

# Lấy trang hiện tại từ session_state
page_selection = st.session_state.page_selection

# =============================================================================
# --- TRANG 1: DỰ BÁO TRỰC TIẾP ---
# =============================================================================
if page_selection == "Forecasting":
    # st.title("☀️ Dự báo Nhiệt độ Hà Nội")
    # st.markdown("Trang này hiển thị kết quả dự báo mới nhất và cho phép bạn chạy lại quy trình.")

    # --- PHẦN MỚI: HIỂN THỊ THỜI TIẾT HIỆN TẠI TỪ API ---
    # st.subheader("Thời tiết hiện tại ở Hà Nội")
    
    realtime_data = fetch_realtime_weather("Hanoi", api_keys=load_keys_from_env())

    if realtime_data:
        # Tạo HTML trực tiếp thay vì dùng st.markdown riêng lẻ
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Chọn icon phù hợp
            if realtime_data.get("chance_of_rain", 0) > 50:
                icon_path = r'assets/heavy-rain.png'
            elif realtime_data.get("wind_speed", 0) > 20:
                icon_path = r'assets/wind.png'
            elif datetime.now().hour >= 18 or datetime.now().hour < 6:
                icon_path = r'assets/moon.png'
            else:
                icon_path = PATH_WEATHER_ICON
            
            # Lấy thông tin ngày tháng
            day_of_week = datetime.now().strftime("%A")  # Thứ trong tuần
            date_time = datetime.now().strftime("%d %B %Y")  # Ngày tháng năm
            
            # Lấy mô tả thời tiết
            weather_condition = realtime_data.get("conditions", "Unknown")
            
            # Tạo HTML block hoàn chỉnh
            real_time_main_html = f"""
            <div class="main-info-block">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <p class="city-name">📍 Ha Noi</p>
                        <p class="day-of-week">{day_of_week}</p>
                        <p class="date-time">{date_time}</p>
                        <p class="big-temp">{int(realtime_data.get("temperature", 0))}°C</p>
                    </div>
                    <div class="weather-icon-wrapper">
                        <img src="data:image/png;base64,{get_img_as_base64(icon_path)}" alt="Weather icon">
                        <p class="weather-condition">{weather_condition}</p>
                    </div>
                </div>
            </div>
            """
            st.markdown(real_time_main_html, unsafe_allow_html=True)
        
        with col2:
            # Tạo block thông tin chi tiết
            weather_details_html = f"""
            <div class="weather-details-block">
                <div class="detail-grid">
                    <div class="detail-item">
                        <p class="detail-label">😬Feels Like</p>
                        <p class="detail-value">{realtime_data.get("feels_like", 0):.1f}°</p>
                    </div>
                    <div class="detail-item">
                        <p class="detail-label">☀️UV Index</p>
                        <p class="detail-value">{realtime_data.get("uv_index", 0)}</p>
                    </div>
                    <div class="detail-item">
                        <p class="detail-label">💨Wind Speed</p>
                        <p class="detail-value">{realtime_data.get("wind_speed", 0):.1f} km/h</p>
                    </div>
                    <div class="detail-item">
                        <p class="detail-label">💧Humidity</p>
                        <p class="detail-value">{realtime_data.get("humidity", 0):.0f}%</p>
                </div>
            </div>
            """
            st.markdown(weather_details_html, unsafe_allow_html=True)
        
    else:
        st.warning("Không thể tải dữ liệu thời tiết hiện tại. Vui lòng kiểm tra lại cấu hình.")

    # --- HIỂN THỊ DỰ BÁO CỦA MÔ HÌNH ---
    predictions_df = load_csv(PATH_PREDICTIONS)
    
    if predictions_df is not None and not predictions_df.empty:
        latest_forecast = predictions_df.iloc[-1]
        forecast_date = pd.to_datetime(latest_forecast['date'])
        forecast_values = latest_forecast[[f"pred_day_{i}" for i in range(1, 6)]].values
        forecast_dates = [forecast_date + timedelta(days=i) for i in range(1, 6)]
        
        # TẠO HTML CHO CÁC CARD DỰ BÁO - FIX: Loại bỏ textwrap.dedent ở đây
        forecast_cards_html = ""
        for date, temp in zip(forecast_dates, forecast_values):
            day_name = date.strftime("%a")
            date_str = date.strftime("%d/%m")
            # Không dùng textwrap.dedent cho từng card
            forecast_cards_html += f'<div class="forecast-card"><p class="forecast-day">{day_name}</p><p class="forecast-date">{date_str}</p><p class="forecast-temp">{temp:.0f}°</p></div>'
        
        # TẠO KHỐI HTML HOÀN CHỈNH
        forecast_html_block = f"""
<div class="forecast-block">
    <p class="forecast-title">🔮 5-Day Temperature Forecast (Model)</p>
    <div class="forecast-cards">
        {forecast_cards_html}
    </div>
</div>
"""
        st.markdown(forecast_html_block, unsafe_allow_html=True)
        
        # BIỂU ĐỒ
        chart_data = pd.DataFrame({'Date': forecast_dates, 'Temperature (°C)': forecast_values}).set_index('Date')
        st.line_chart(chart_data, use_container_width=True, height=250)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # NÚT CẬP NHẬT
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Update & Run Model Forecast Again", use_container_width=True):
                with st.spinner("Processing..."):
                    try:
                        daily_update() 
                        st.success("✅ Forecast updated successfully!")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error during forecast: {e}")
        
    else:
        st.warning(f"⚠️ Không tìm thấy dữ liệu dự báo của mô hình tại '{PATH_PREDICTIONS}'.")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Chạy Dự báo của Mô hình lần đầu", use_container_width=True):
                with st.spinner("Running first-time forecast..."):
                    try:
                        daily_update()
                        st.success("✅ Initial forecast completed!")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")


# =============================================================================
# --- TRANG 2: PHÂN TÍCH DỮ LIỆU LỊCH SỬ ---
# =============================================================================
elif page_selection == "Historical Data Analysis":
    st.title("📊 Phân tích Dữ liệu Lịch sử")
    st.markdown("Khám phá dữ liệu được sử dụng để huấn luyện mô hình.")
    
    st.header("Khám phá Dữ liệu Thời tiết trong 3 năm gần nhất")

    df_3y = load_csv(PATH_3_YEAR_DATA)

    if df_3y is not None:
        df_3y['datetime'] = pd.to_datetime(df_3y['datetime'])

        st.subheader("Xu hướng Nhiệt độ Trung bình (3 năm)")
        st.line_chart(df_3y.set_index('datetime')['temp'])

        st.subheader("Ma trận Tương quan giữa các Đặc trưng")
        st.info("Biểu đồ này cho thấy mối quan hệ tuyến tính giữa các yếu tố thời tiết. Màu càng gần +1 (đỏ) hoặc -1 (xanh) cho thấy tương quan càng mạnh.")

        numeric_cols = df_3y.select_dtypes(include=['number']).columns
        corr = df_3y[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(corr, ax=ax, cmap='coolwarm', annot=False)
        st.pyplot(fig)

        if st.checkbox("Hiển thị Dữ liệu Thô (Raw Data)"):
            st.dataframe(df_3y)
    else:
        st.error(f"Không tìm thấy file dữ liệu tại '{PATH_3_YEAR_DATA}'.")


# =============================================================================
# --- TRANG 3: GIÁM SÁT HIỆU SUẤT MÔ HÌNH ---
# =============================================================================
elif page_selection == "Model Performance":
    st.title("⚙️ Giám sát Hiệu suất Mô hình")
    st.markdown("Theo dõi và đánh giá độ chính xác của mô hình theo thời gian.")
    
    st.header("Theo dõi và Đánh giá Độ chính xác của Mô hình")

    st.subheader("Lịch sử lỗi RMSE theo thời gian")
    rmse_logs = load_joblib(PATH_RMSE_LOG)
    if rmse_logs is not None:
        df_rmse = pd.DataFrame(rmse_logs)
        df_rmse['base_date'] = pd.to_datetime(df_rmse['base_date'])
        st.line_chart(df_rmse.set_index('base_date')['rmse'].dropna())
        st.caption("Xu hướng lỗi tăng dần có thể là dấu hiệu mô hình cần được huấn luyện lại.")
    else:
        st.warning(f"Không tìm thấy file log RMSE tại '{PATH_RMSE_LOG}'.")

    st.markdown("---")

    st.subheader("So sánh giữa Dự báo và Thực tế")
    pred_df_comp = load_csv(PATH_PREDICTIONS)
    actual_df_comp = load_csv(PATH_RAW_3WEEKS)

    if pred_df_comp is not None and actual_df_comp is not None:
        pred_df_comp['date'] = pd.to_datetime(pred_df_comp['date'])
        actual_df_comp['datetime'] = pd.to_datetime(actual_df_comp['datetime'])

        available_dates = pred_df_comp['date']
        selected_date = st.selectbox(
            "Chọn một ngày dự báo trong quá khứ để so sánh:",
            options=available_dates,
            format_func=lambda date: date.strftime('%Y-%m-%d'),
            index=len(available_dates) - 1 if not available_dates.empty else 0
        )

        selected_row = pred_df_comp[pred_df_comp['date'] == selected_date]

        if not selected_row.empty:
            forecast_dates = [selected_date + timedelta(days=i) for i in range(1, 6)]
            forecast_values = selected_row.iloc[0][[f'pred_day_{i}' for i in range(1, 6)]].values

            actual_values = []
            for d in forecast_dates:
                val = actual_df_comp.loc[actual_df_comp['datetime'].dt.date == d.date(), 'temp']
                actual_values.append(val.values[0] if not val.empty else None)

            comparison_df = pd.DataFrame({
                'Ngày': forecast_dates,
                'Dự báo': forecast_values,
                'Thực tế': actual_values
            }).set_index('Ngày')

            st.line_chart(comparison_df)
            st.table(comparison_df)
        else:
            st.warning("Không có nhật ký dự báo nào cho ngày đã chọn.")
    else:
        st.warning(f"Không tìm thấy file '{PATH_PREDICTIONS}' hoặc '{PATH_RAW_3WEEKS}' để so sánh.")

    st.markdown("---")

    st.subheader("Lịch sử Huấn luyện lại Mô hình")
    retrain_logs = load_joblib(PATH_RETRAIN_LOG)
    if retrain_logs:
        for record in reversed(retrain_logs):
            with st.expander(f"Lần huấn luyện lại vào lúc {record['timestamp']}"):
                st.metric("RMSE trung bình đạt được", f"{record['metrics']['average']['RMSE']:.4f}")
                st.write("Các siêu tham số tốt nhất:")
                st.json(record['best_params'], expanded=False)
    else:
        st.info("Chưa có lịch sử huấn luyện lại nào được ghi nhận.")