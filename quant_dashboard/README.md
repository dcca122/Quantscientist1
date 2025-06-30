# Quant Dashboard

This Streamlit application provides an interactive interface for running a simple quant strategy using techniques from the repository.

## Features

- Download price data from Yahoo Finance
- Compute technical indicators (SMA, RSI, ATR)
- Detect market regimes using a Hidden Markov Model
- Train a logistic regression model to predict next day returns
- Backtest the resulting signals and display performance metrics
- Visualize price series, signals, drawdowns and feature importances
- Upload portfolio history to view cumulative returns, Sharpe ratio and
  drawdowns
- Scan tickers for upcoming earnings and display volatility metrics

## Usage

```bash
pip install -r ../../requirements-minimal.txt
streamlit run app.py
```
