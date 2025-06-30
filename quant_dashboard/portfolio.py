import pandas as pd
import numpy as np


def compute_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Compute risk statistics for a portfolio value series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``Date`` column and ``Value`` column representing
        portfolio value over time.

    Returns
    -------
    stats : pd.DataFrame
        Table of cumulative return, annualized return, Sharpe ratio and
        maximum drawdown.
    equity_curve : pd.Series
        Cumulative growth of $1 invested in the portfolio.
    drawdown : pd.Series
        Drawdown series.
    """
    df = df.sort_values("Date").set_index("Date")
    returns = df["Value"].pct_change().dropna()
    equity_curve = (1 + returns).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1

    ann_return = equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1 if len(equity_curve) > 0 else 0
    sharpe = np.sqrt(252) * returns.mean() / returns.std(ddof=0) if returns.std(ddof=0) != 0 else np.nan
    stats = pd.DataFrame({
        "Cumulative Return": equity_curve.iloc[-1] - 1,
        "Annualized Return": ann_return,
        "Sharpe": sharpe,
        "Max Drawdown": drawdown.min(),
    }, index=[0])

    return stats, equity_curve, drawdown
