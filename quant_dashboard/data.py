import pandas as pd
import yfinance as yf


def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    data = yf.download(ticker, start=start, end=end, progress=False)
    data.index.name = "Date"
    return data
