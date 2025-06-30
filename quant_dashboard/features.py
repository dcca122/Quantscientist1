import pandas as pd
import numpy as np


def sma(df: pd.DataFrame, window: int, column: str = "Close") -> pd.Series:
    """Simple moving average."""
    return df[column].rolling(window).mean()


def rsi(df: pd.DataFrame, window: int = 14, column: str = "Close") -> pd.Series:
    """Relative Strength Index."""
    delta = df[column].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(gain).rolling(window).mean()
    roll_down = pd.Series(loss).rolling(window).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range."""
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators used in the strategy."""
    df = df.copy()
    df["sma_fast"] = sma(df, 20)
    df["sma_slow"] = sma(df, 50)
    df["rsi"] = rsi(df)
    df["atr"] = atr(df)
    df.dropna(inplace=True)
    return df
