import pandas as pd
import numpy as np


def backtest(prices: pd.Series, signals: pd.Series) -> pd.DataFrame:
    """Simple backtest applying signals to price returns."""
    returns = prices.pct_change().fillna(0)
    strategy_returns = returns * signals.shift(1).fillna(0)
    cum = (1 + strategy_returns).cumprod()
    dd = cum / cum.cummax() - 1
    stats = {
        "Cumulative Return": cum.iloc[-1] - 1,
        "Sharpe": np.sqrt(252) * strategy_returns.mean() / strategy_returns.std(ddof=0),
        "Max Drawdown": dd.min(),
        "Win Rate": (strategy_returns > 0).mean(),
    }
    return pd.DataFrame(stats, index=[0]), cum, dd
