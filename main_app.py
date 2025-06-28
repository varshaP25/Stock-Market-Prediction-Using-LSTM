import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from textblob import TextBlob
from datetime import datetime
import math
import os
import time
from streamlit_autorefresh import st_autorefresh

# --- Setup ---
st.set_page_config(page_title="Integrated Stock App", layout="wide")
st.title(" Stock Market Dashboard")

# --- Theme toggle ---
dark_mode = st.sidebar.checkbox(" Dark Mode")
plt.style.use('dark_background' if dark_mode else 'ggplot')

# --- Load Model ---
MODEL_PATH = "Stock Predictions Model.keras"
model = tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# --- User Input ---
stock = st.text_input("Enter Stock Symbol", "AAPL")
start = "2022-01-01"
end = datetime.now().strftime("%Y-%m-%d")

# --- Fetch Stock Data ---
@st.cache_data(ttl=3600)
def load_data(symbol, start, end):
    return yf.download(symbol, start, end)

data = load_data(stock, start, end)

# --- Moving Averages ---
data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()
data["MA100"] = data["Close"].rolling(100).mean()
data["MA200"] = data["Close"].rolling(200).mean()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Charts & Prediction", " Metrics", " News Sentiment"])

# --- Tab 1: Charts & Prediction ---
with tab1:
    st.subheader("Price Chart with Moving Averages")
    fig, ax = plt.subplots()
    data[['Close', 'MA20', 'MA50', 'MA100', 'MA200']].plot(ax=ax, title=f"{stock} Stock Prices")
    st.pyplot(fig)

    st.subheader(" Next Price Prediction")
    if model is not None and len(data) >= 100:
        recent_data = data['Close'].tail(100).values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(recent_data)
        x_pred = np.array([scaled_data[i-100:i] for i in range(100, len(scaled_data)+1)])
        if len(x_pred) > 0:
            pred_price = model.predict(x_pred)[-1][0] * (1 / scaler.scale_[0])
            st.success(f" Predicted next price: **{pred_price:.2f}**")
    else:
        st.warning("Not enough data for prediction or model not found.")

    # Trend Summary
    st.subheader("📊 Trend Summary")
    if not data["MA20"].isna().all() and not data["MA50"].isna().all():
        ma20, ma50 = data["MA20"].iloc[-1], data["MA50"].iloc[-1]
        if ma20 > ma50:
            st.success("📈 Uptrend (MA20 > MA50)")
        elif ma20 < ma50:
            st.error("📉 Downtrend (MA20 < MA50)")
        else:
            st.info("➖ Sideways trend")

# --- Tab 2: Metrics from Full Prediction ---
with tab2:
    st.subheader("🔁 Full Prediction Metrics (Last 20%)")

    if model is not None and len(data) > 120:
        data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
        data_test = pd.DataFrame(data.Close[int(len(data)*0.80):])
        scaler = MinMaxScaler()
        pas_100_days = data_train.tail(100)
        data_test_comb = pd.concat([pas_100_days, data_test], ignore_index=True)
        scaled = scaler.fit_transform(data_test_comb)

        x, y_true = [], []
        for i in range(100, scaled.shape[0]):
            x.append(scaled[i-100:i])
            y_true.append(scaled[i, 0])
        x, y_true = np.array(x), np.array(y_true)

        pred = model.predict(x)
        pred *= (1 / scaler.scale_[0])
        y_true *= (1 / scaler.scale_[0])

        mse = mean_squared_error(y_true, pred)
        mae = mean_absolute_error(y_true, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, pred)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📉 MAE", f"{mae:.2f}")
        col2.metric("📉 MSE", f"{mse:.2f}")
        col3.metric("📉 RMSE", f"{rmse:.2f}")
        col4.metric("📈 R²", f"{r2:.2f}")

        fig2 = plt.figure(figsize=(8,6))
        plt.plot(y_true, 'r', label='Original')
        plt.plot(pred, 'g', label='Predicted')
        plt.legend()
        plt.title("Original vs Predicted Prices")
        st.pyplot(fig2)
    else:
        st.warning("Not enough data for metrics calculation.")

# --- Tab 3: News Sentiment ---
with tab3:
    st.subheader("🗞️ Simulated News Sentiment")

    def get_mock_news_sentiment():
        news = [
            "Stock price hits all-time high",
            "CEO resigns due to internal conflicts",
            "New product line receives positive reviews",
            "Market uncertainty increases volatility"
        ]
        return [(n, TextBlob(n).sentiment.polarity) for n in news]

    for headline, polarity in get_mock_news_sentiment():
        icon = "🔺" if polarity > 0 else "🔻" if polarity < 0 else "⚖️"
        st.markdown(f"- {headline} — Sentiment: {icon} ({polarity:.2f})")

# --- Auto Refresh Logic ---
refresh_interval = 60  # seconds
remaining = refresh_interval - ((time.time() - st.session_state.get("last_run", time.time())) % refresh_interval)
st.info(f"Auto-refresh in {int(remaining)}s …")
if remaining < 1:
    st.session_state.last_run = time.time()
    st.rerun()
st_autorefresh(interval=60_000, key="live_refresh")
