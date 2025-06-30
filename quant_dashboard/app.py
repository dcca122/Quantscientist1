import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from .data import load_data
from .features import add_features
from .regime import detect_regime
from .model import train_model, generate_signals
from .backtest import backtest


st.title("Quant Dashboard")

with st.sidebar:
    ticker = st.text_input("Ticker", value="SPY")
    start = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
    end = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))
    threshold = st.slider("Signal Threshold", 0.5, 0.7, value=0.55, step=0.01)
    run = st.button("Run Backtest")

if run:
    data = load_data(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    data = add_features(data)
    data["regime"] = detect_regime(data)
    model, train_info, importance = train_model(data)
    signals = generate_signals(model, data, threshold)
    stats, cum, dd = backtest(data["Close"], signals)

    st.subheader("Price and Signals")
    fig, ax = plt.subplots()
    ax.plot(data.index, data["Close"], label="Close")
    ax.plot(data.index, data["sma_fast"], label="SMA20", alpha=0.5)
    ax.plot(data.index, data["sma_slow"], label="SMA50", alpha=0.5)
    ax.scatter(data.index[signals == 1], data["Close"][signals == 1], marker="^", color="green")
    ax.scatter(data.index[signals == -1], data["Close"][signals == -1], marker="v", color="red")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Backtest Performance")
    st.table(stats)
    fig2, ax2 = plt.subplots()
    ax2.plot(cum.index, cum, label="Equity Curve")
    ax2.fill_between(dd.index, dd, color="red", alpha=0.3, label="Drawdown")
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("Feature Importance")
    st.bar_chart(importance.sort_values())

    st.subheader("Model Prediction Accuracy")
    st.write(train_info["actual"].eq(train_info["prob"] > 0.5).mean())
