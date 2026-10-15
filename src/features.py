"""Пользовательский трансформер для крипто-признаков (ПР-3)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CryptoFeatures(BaseEstimator, TransformerMixin):
    """Признаки дня t из OHLCV-данных строго до дня t (лаг >= 1 — без утечки)."""

    def fit(self, X, y=None):
        return self                      # stateless: запоминать нечего

    def transform(self, X):
        X = pd.DataFrame(X)
        close = X["close"]
        log_ret = np.log(close / close.shift(1))       # доходность дня

        out = pd.DataFrame(index=X.index)
        for lag in [1, 2, 3, 5]:
            out[f"ret_lag{lag}"] = log_ret.shift(lag)  # доходности t-1, t-2, ...
        rng = (X["high"] - X["low"]) / X["close"] * 100
        out["range_lag1"] = rng.shift(1)
        out["volatility_7"] = log_ret.rolling(7).std().shift(1)
        out["vol_ratio"] = X["quote_volume"] / X["quote_volume"].shift(1)
        return out