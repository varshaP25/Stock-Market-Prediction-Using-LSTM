# intraday_lstm_training.py

import os
import time
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from alpha_vantage.timeseries import TimeSeries

# --- Configuration ---
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "56Z0NNV2516MH28J")
INTERVAL = "1min"
MODEL_SAVE_DIR = "models"
LOG_PATH = "prediction_log.csv"


# --- Streamlit UI ---
def main():
    st.set_page_config(page_title=" Intraday Dashboard", layout="wide")
    auto_refresh()

    apply_custom_style()
    st.title(" Intraday Stock Trend Prediction Dashboard")

    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, RELIANCE.BSE):", value="AMZN").upper()
    model_path = os.path.join(MODEL_SAVE_DIR, f"{symbol}_model.keras")

    model, scaler = load_or_train_model(symbol, model_path)

    df = fetch_intraday_data(symbol)
    plot_intraday_prices(df, symbol)

    try:
        x_pred = prepare_lstm_prediction_data(df, scaler)
        predicted_price = scaler.inverse_transform(model.predict(x_pred))[0][0]
    except ValueError as ve:
        st.error(f"Prediction Error: {ve}")
        return

    actual_price = df['Close'].iloc[-1]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_prediction(timestamp, actual_price, predicted_price)

    display_price_metrics(actual_price, predicted_price)
    display_prediction_log()

    st.caption("🔁 Auto-refresh every 60 seconds. Powered by LSTM & Alpha Vantage API.")


# --- Auto Refresh ---
def auto_refresh():
    if "last_run" not in st.session_state:
        st.session_state.last_run = time.time()
    if time.time() - st.session_state.last_run > 60:
        st.session_state.last_run = time.time()
        st.experimental_rerun()


# --- Style ---
def apply_custom_style():
    st.markdown("""
        <style>
        .main, .stApp {
            background-color: #f0f2f6;
        }
        .stApp {
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)


# --- Data Fetch ---
def fetch_intraday_data(symbol, interval=INTERVAL, outputsize="compact"):
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    data, _ = ts.get_intraday(symbol=symbol, interval=interval, outputsize=outputsize)
    data = data.rename(columns=lambda x: x.split('. ')[1].capitalize())
    data.index = pd.to_datetime(data.index)
    return data.sort_index()


# --- LSTM Data Prep ---
def prepare_lstm_data(data, window_size=100):
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)
    x, y = [], []
    for i in range(window_size, len(scaled)):
        x.append(scaled[i - window_size:i])
        y.append(scaled[i])
    return np.array(x), np.array(y), scaler


def prepare_lstm_prediction_data(data, scaler, window_size=100):
    close_prices = data['Close'].values.reshape(-1, 1)
    if len(close_prices) < window_size:
        raise ValueError(f"Not enough data. Need at least {window_size}, got {len(close_prices)}.")
    scaled = scaler.transform(close_prices)
    return np.array([scaled[-window_size:]])


# --- Model ---
def train_lstm_model(x, y, epochs=10, batch_size=32):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(x.shape[1], 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model


def load_or_train_model(symbol, model_path):
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    try:
        if os.path.exists(model_path):
            model = load_model(model_path)
            st.success("Model loaded successfully.")
        else:
            st.warning(f"Model for {symbol} not found. Training a new model...")
            df_train = fetch_intraday_data(symbol, outputsize="full")
            x, y, scaler = prepare_lstm_data(df_train)
            model = train_lstm_model(x, y)
            model.save(model_path)
            st.success("Model trained and saved.")
        # Ensure scaler is fitted regardless of training or loading
        df_latest = fetch_intraday_data(symbol)
        _, _, scaler = prepare_lstm_data(df_latest)
        return model, scaler
    except Exception as e:
        st.error(f"Model Error: {e}")
        st.stop()


# --- Visualization ---
def plot_intraday_prices(df, symbol):
    st.header(f" Latest Intraday Data for {symbol}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title=f"Real-Time Stock Prices ({symbol})", xaxis_title='Time', yaxis_title='Price (₹)')
    st.plotly_chart(fig, use_container_width=True)


def display_price_metrics(actual, predicted):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(" Actual Price")
        st.metric("Current Close Price", f"₹{actual:.2f}")
    with col2:
        st.subheader(" Predicted Price")
        st.metric("Next Minute Prediction", f"₹{predicted:.2f}")


def display_prediction_log():
    if os.path.exists(LOG_PATH):
        log_df = pd.read_csv(LOG_PATH).tail(100)
        st.subheader("📈 Prediction vs Actual Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=log_df['Time'], y=log_df['Actual'], mode='lines+markers', name='Actual'))
        fig.add_trace(go.Scatter(x=log_df['Time'], y=log_df['Predicted'], mode='lines+markers', name='Predicted'))
        fig.update_layout(xaxis_title='Time', yaxis_title='Price (₹)', title='Prediction vs Actual')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(" Download Prediction Log")
        with open(LOG_PATH, "rb") as f:
            st.download_button("Download as CSV", data=f, file_name="prediction_log.csv", mime="text/csv")


# --- Logging ---
def log_prediction(timestamp, actual, predicted):
    new_entry = pd.DataFrame({'Time': [timestamp], 'Actual': [actual], 'Predicted': [predicted]})
    if os.path.exists(LOG_PATH):
        log_df = pd.read_csv(LOG_PATH)
        log_df = pd.concat([log_df, new_entry], ignore_index=True)
    else:
        log_df = new_entry
    log_df.to_csv(LOG_PATH, index=False)


if __name__ == "__main__":
    main()
