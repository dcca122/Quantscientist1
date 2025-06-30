import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_model(df: pd.DataFrame) -> tuple[LogisticRegression, pd.DataFrame]:
    """Train logistic regression to predict next-day returns sign."""
    X = df[["sma_fast", "sma_slow", "rsi", "atr", "regime"]]
    y = (df["Close"].shift(-1) > df["Close"]).astype(int)  # next-day up/down
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    preds = model.predict_proba(X)[:, 1]
    accuracy = accuracy_score(y, preds > 0.5)
    importance = pd.Series(model.coef_[0], index=X.columns)
    return model, pd.DataFrame({"prob": preds, "actual": y}, index=X.index), importance


def generate_signals(model: LogisticRegression, df: pd.DataFrame, threshold: float = 0.55) -> pd.Series:
    """Return trading signals based on model probabilities."""
    X = df[["sma_fast", "sma_slow", "rsi", "atr", "regime"]]
    probs = model.predict_proba(X)[:, 1]
    signals = pd.Series(0, index=df.index)
    signals[probs > threshold] = 1
    signals[probs < 1 - threshold] = -1
    return signals
