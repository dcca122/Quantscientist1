import pandas as pd
import numpy as np
import yfinance as yf


def _term_structure_slope(ticker: str) -> float | None:
    """Return slope between front month and 45-day implied vol using options data.

    This function relies on yfinance and will return ``None`` if data
    cannot be fetched.
    """
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return None
        # find nearest and 45d expirations
        dates = pd.to_datetime(expirations)
        today = pd.Timestamp.today().normalize()
        front = dates[dates >= today]
        if front.empty:
            return None
        near_exp = front.min()
        target = today + pd.Timedelta(days=45)
        near_45 = front[(front - target).abs().argsort()].iloc[0]

        def iv_atm(exp):
            opt = tk.option_chain(exp)
            calls = opt.calls
            if calls.empty:
                return None
            atm = calls.iloc[(calls["strike"] - tk.history(period="1d")["Close"].iloc[-1]).abs().argmin()]
            return atm.get("impliedVolatility")

        iv_near = iv_atm(near_exp.strftime("%Y-%m-%d"))
        iv_45 = iv_atm(near_45.strftime("%Y-%m-%d"))
        if iv_near is None or iv_45 is None:
            return None
        return iv_near - iv_45
    except Exception:
        return None


def _avg_volume(ticker: str) -> float | None:
    """Return 30-day average volume prior to today."""
    try:
        hist = yf.download(ticker, period="60d", progress=False)
        return hist["Volume"].iloc[-30:].mean()
    except Exception:
        return None


def _iv30_rv30(ticker: str) -> float | None:
    """Return ratio of 30-day implied volatility to realized volatility."""
    try:
        tk = yf.Ticker(ticker)
        iv = tk.info.get("impliedVolatility")
        if iv is None:
            return None
        hist = tk.history(period="60d")
        rv = hist["Close"].pct_change().rolling(30).std().iloc[-1] * np.sqrt(252)
        return iv / rv if rv != 0 else None
    except Exception:
        return None


def scan(tickers: list[str]) -> pd.DataFrame:
    """Scan tickers for upcoming earnings and compute metrics."""
    rows = []
    for t in tickers:
        slope = _term_structure_slope(t)
        vol = _avg_volume(t)
        ratio = _iv30_rv30(t)
        if None in (slope, vol, ratio):
            rec = "Avoid"
        elif slope < 0 and ratio > 1 and vol > 0:
            rec = "Recommended"
        elif slope < 0 or ratio > 1:
            rec = "Consider"
        else:
            rec = "Avoid"
        rows.append({
            "Ticker": t,
            "Term Structure Slope": slope,
            "Avg Volume": vol,
            "IV30/RV30": ratio,
            "Recommendation": rec,
        })
    return pd.DataFrame(rows)
