import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def detect_regime(df: pd.DataFrame, n_states: int = 2) -> pd.Series:
    """Fit a Hidden Markov Model on returns and return regime labels."""
    returns = np.log(df["Close"]).diff().dropna().values.reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100)
    model.fit(returns)
    regimes = model.predict(returns)
    regime_series = pd.Series(regimes, index=df.index[1:], name="regime")
    # Forward fill the first value
    regime_series = regime_series.reindex(df.index, method="ffill").fillna(0)
    return regime_series
