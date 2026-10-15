"""Тесты трансформера CryptoFeatures (ПР-3)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # корень проекта

from src.features import CryptoFeatures


def make_raw(n=30, seed=0):
    """Искусственные OHLCV-данные с известными значениями."""
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
    high = close * (1 + abs(rng.normal(0, 0.01, n)))
    low = close * (1 - abs(rng.normal(0, 0.01, n)))
    quote_volume = rng.uniform(1e6, 5e6, n)
    return pd.DataFrame({"high": high, "low": low, "close": close,
                         "quote_volume": quote_volume})


def test_columns_and_shape():
    """Тест 1: трансформер выдаёт ровно те 7 колонок, что задуманы."""
    Xt = CryptoFeatures().fit_transform(make_raw())
    assert list(Xt.columns) == ["ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5",
                                "range_lag1", "volatility_7", "vol_ratio"]
    assert Xt.shape == (30, 7)


def test_lag_values_correct():
    """Тест 2: лаги сдвинуты правильно — сверяем со сдвигом, посчитанным вручную."""
    X = make_raw()
    Xt = CryptoFeatures().fit_transform(X)
    log_ret = np.log(X["close"] / X["close"].shift(1))
    pd.testing.assert_series_equal(
        Xt["ret_lag1"].reset_index(drop=True),
        log_ret.shift(1).reset_index(drop=True),
        check_names=False,
    )


def test_stateless_fit():
    """Тест 3: класс stateless — fit на других данных не меняет результат."""
    X = make_raw(seed=1)
    tr = CryptoFeatures()
    out1 = tr.fit(X).transform(X)
    out2 = tr.fit(X.iloc[:5]).transform(X)   # fit на подмножестве
    pd.testing.assert_frame_equal(out1, out2)