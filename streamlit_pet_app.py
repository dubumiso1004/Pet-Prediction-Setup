# streamlit_pet_app.py (Cloud 호환 .joblib 파일 로딩으로 수정됨)
import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
import requests
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime
import os

plt.rcParams["font.family"] = "Malgun Gothic"
rcParams.update({'axes.titlesize': 12, 'axes.labelsize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8})

def dms_to_decimal(dms_str):
    try:
        d, m, s = map(float, dms_str.split(";"))
        return d + m / 60 + s / 3600
    except:
        return None

@st.cache_data
def load_data():
    df = pd.read_excel("total_svf_gvi_bvi_250618.xlsx", sheet_name="gps 포함")
    df.columns = df.columns.str.strip().str.lower()
    df["lat_decimal"] = df["lat"].apply(dms_to_decimal)
    df["lon_decimal"] = df["lon"].apply(dms_to_decimal)
    return df

# ✅ Cloud 호환 .joblib 파일 사용
model = joblib.load("pet_rf_model_trained.joblib")
df = load_data()
LOG_FILE = "pet_prediction_log.csv"
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp", "lat", "lon", "SVF", "GVI", "BVI", "Temp", "Humidity", "Wind", "PET", "PET_future", "PET_selected"]).to_csv(LOG_FILE, index=False)

st.set_page_config(page_title="PET 예측 지도", layout="wide")
st.title("📍 지도 기반 PET 예측 시스템 + 미래 예측")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("### 🗺️ 위치 선택")
    m = folium.Map(location=[35.2321, 129.0790], zoom_start=17)
    click = st_folium(m, height=500)

with col2:
    if click and click.get("last_clicked"):
        lat = click["last_clicked"]["lat"]
        lon = click["last_clicked"]["lng"]

        st.markdown("### 📌 선택 위치")
        st.write(f"위도: {lat:.5f}, 경도: {lon:.5f}")

        df["distance"] = ((df["lat_decimal"] - lat)**2 + (df["lon_decimal"] - lon)**2)**0.5
        nearest = df.loc[df["distance"].idxmin()]

        st.markdown("#### 🎛️ 시각지표 조절")
        svf = st.slider("SVF (하늘 비율)", 0.0, 1.0, float(nearest["svf"]), 0.01)
        gvi = st.slider("GVI (녹지 비율)", 0.0, 1.0, float(nearest["gvi"]), 0.01)
        bvi = st.slider("BVI (건물 비율)", 0.0, 1.0, float(nearest["bvi"]), 0.01)

        try:
            api_key = "2ced117aca9b446ae43cf82401d542a8"
            url_now = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            url_forecast = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            now_res = requests.get(url_now).json()
            forecast_res = requests.get(url_forecast).json()

            air_temp = now_res["main"]["temp"]
            humidity = now_res["main"]["humidity"]
            wind_speed = now_res["wind"]["speed"]

            forecast_list = forecast_res["list"]
            time_options = [entry["dt_txt"] for entry in forecast_list]
            selected_time = st.selectbox("🌤 예측 시간 선택 (3시간 간격)", time_options)
            selected = next(e for e in forecast_list if e["dt_txt"] == selected_time)
            f_temp = selected["main"]["temp"]
            f_humidity = selected["main"]["humidity"]
            f_wind = selected["wind"]["speed"]
        except:
            air_temp = nearest["airtemperature"]
            humidity = nearest["humidity"]
            wind_speed = nearest["windspeed"]
            f_temp = air_temp
            f_humidity = humidity
            f_wind = wind_speed
            selected_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        now_input = pd.DataFrame([{
            "SVF": svf, "GVI": gvi, "BVI": bvi,
            "AirTemperature": air_temp,
            "Humidity": humidity,
            "WindSpeed": wind_speed
        }])
        pet_now = model.predict(now_input)[0]

        future_input = pd.DataFrame([{
            "SVF": svf, "GVI": gvi, "BVI": bvi,
            "AirTemperature": f_temp,
            "Humidity": f_humidity,
            "WindSpeed": f_wind
        }])
        pet_future = model.predict(future_input)[0]

        st.markdown("### 🤖 예측된 PET")
        st.success(f"현재 PET: **{pet_now:.2f}°C**, 선택시간({selected_time}) PET: **{pet_future:.2f}°C**")

        log = pd.DataFrame([{
            "timestamp": datetime.now(),
            "lat": lat, "lon": lon,
            "SVF": svf, "GVI": gvi, "BVI": bvi,
            "Temp": air_temp, "Humidity": humidity, "Wind": wind_speed,
            "PET": pet_now, "PET_future": pet_future,
            "PET_selected": selected_time
        }])
        log.to_csv(LOG_FILE, mode="a", header=False, index=False)

        st.markdown("### 📊 PET 변화 추이")
        hist = pd.read_csv(LOG_FILE)
        recent = hist[(abs(hist["lat"] - lat) < 0.0001) & (abs(hist["lon"] - lon) < 0.0001)]
        if not recent.empty:
            recent = recent.sort_values("timestamp")
            time_now = pd.to_datetime(recent["timestamp"])
            time_future = pd.to_datetime(recent["PET_selected"], errors='coerce')
            fig, ax = plt.subplots()
            ax.plot(time_now, recent["PET"], marker='o', label="현재 PET")
            ax.plot(time_future, recent["PET_future"], marker='x', linestyle="--", label="선택 시간 PET")
            for i in range(len(recent)):
                ax.annotate(f"T={recent['Temp'].iloc[i]:.1f}\nRH={recent['Humidity'].iloc[i]:.0f}%\nWS={recent['Wind'].iloc[i]:.1f}m/s",
                            (time_future.iloc[i], recent["PET_future"].iloc[i]),
                            textcoords="offset points", xytext=(0,8), ha='center', fontsize=8, color='gray')
            ax.set_ylabel("PET (°C)")
            ax.set_xlabel("시간")
            ax.set_title("PET 변화 추이 (현재 vs 선택 시간)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            with st.expander("🔍 변화 원인 상세 보기"):
                st.dataframe(recent[["timestamp", "PET", "PET_future", "Temp", "Humidity", "Wind", "SVF", "GVI", "BVI"]])
    else:
        st.info("지도를 클릭해서 위치를 선택하세요.")
