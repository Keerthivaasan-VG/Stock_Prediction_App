import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import tempfile
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="📈 Stock Price Prediction (LSTM)",
    page_icon="📊",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
h1, h2, h3 {
    color: #00ffd5;
}
.stButton>button {
    background-color: #00ffd5;
    color: black;
    font-weight: bold;
    border-radius: 10px;
}
.stFileUploader {
    background-color: #1e1e1e;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("📈 LSTM-Based Stock Price Prediction")
st.markdown(
    "### 🔮 Predict future stock prices using **Deep Learning (LSTM)**"
)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Settings")

sequence_length = st.sidebar.slider(
    "Sequence Length (Time Steps)",
    min_value=30,
    max_value=120,
    value=60
)

# =========================
# MODEL LOADING
# =========================
@st.cache_resource
def load_lstm_model():
    # 🔗 REPLACE WITH YOUR GOOGLE DRIVE DIRECT DOWNLOAD LINK
    model_url = "https://drive.google.com/file/d/1gF0c2BmEW1wZCJy9CfIyy0ukwErCydbX/view?usp=sharing"

    response = requests.get(model_url)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
    temp_file.write(response.content)
    temp_file.close()

    model = load_model(temp_file.name)
    return model

with st.spinner("🔄 Loading LSTM model..."):
    model = load_lstm_model()

st.success("✅ Model loaded successfully!")

# =========================
# FILE UPLOAD
# =========================
st.subheader("📂 Upload Stock Data (CSV)")

uploaded_file = st.file_uploader(
    "Upload CSV file (must contain 'Close' column)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "Close" not in df.columns:
        st.error("❌ CSV must contain a 'Close' column")
        st.stop()

    st.write("📄 **Data Preview**")
    st.dataframe(df.head())

    # =========================
    # DATA PREPROCESSING
    # =========================
    close_prices = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    X = []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])

    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # =========================
    # PREDICTION
    # =========================
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    actual = close_prices[sequence_length:]

    # =========================
    # VISUALIZATION
    # =========================
    st.subheader("📊 Prediction Results")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual, label="Actual Price", color="cyan")
    ax.plot(predictions, label="Predicted Price", color="magenta")
    ax.set_title("Actual vs Predicted Stock Prices")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()

    st.pyplot(fig)

    # =========================
    # NEXT DAY PREDICTION
    # =========================
    last_sequence = scaled_data[-sequence_length:]
    last_sequence = np.reshape(last_sequence, (1, sequence_length, 1))

    next_day_price = model.predict(last_sequence)
    next_day_price = scaler.inverse_transform(next_day_price)

    st.markdown("## 📌 **Next Day Predicted Price**")
    st.success(f"💰 {next_day_price[0][0]:.2f}")

else:
    st.info("⬆️ Upload a stock CSV file to begin prediction")

# =========================
# FOOTER
# =========================
st.markdown("""
---
👨‍💻 **Built with Streamlit & LSTM**  
📘 Deep Learning for Stock Market Prediction  
""")
