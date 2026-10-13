"""Загрузка полной истории OHLCV с Binance и сборка единого датасета.
Воспроизводимо: повторный запуск ничего не качает заново.
"""
import time
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

COINS = [
    ("btc_full.csv", "BTCUSDT", 1502928000000),  # 17.08.2017 — листинг пары
    ("eth_full.csv", "ETHUSDT", 1502928000000),
    ("sol_full.csv", "SOLUSDT", 1597100000000),  # ~11.08.2020
]

def fetch(symbol, start_ms):
    url = "https://api.binance.com/api/v3/klines"
    rows, cur = [], start_ms
    while True:
        params = {"symbol": symbol, "interval": "1d", "startTime": cur, "limit": 1000}
        data = requests.get(url, params=params, timeout=30).json()
        if not data:
            break
        rows += data
        cur = data[-1][0] + 86_400_000
        time.sleep(0.3)
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base_volume",
            "taker_buy_quote_volume", "ignore"]
    return pd.DataFrame(rows, columns=cols)

for fname, symbol, start_ms in COINS:
    f = RAW / fname
    if f.exists():
        print(f"{fname} уже есть, пропускаю")
        continue
    print(f"Скачиваю {symbol} (полная история)...")
    df = fetch(symbol, start_ms)
    df["symbol"] = symbol
    df.drop(columns=["ignore"]).to_csv(f, index=False)
    print(f"  сохранено {len(df)} строк")

# --- Сборка единого датасета ---
files = [f for f in RAW.glob("*.csv") if f.stat().st_size > 1000]
print(f"Найдено файлов: {len(files)}")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
before = len(df)
df = (df.drop_duplicates(subset=["symbol", "open_time"])
        .sort_values(["symbol", "open_time"], ignore_index=True))
df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.date
df = df.drop(columns=["close_time", "open_time_formatted", "close_time_formatted"],
             errors="ignore")
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "crypto_daily.csv", index=False)
print(f"Строк до/после удаления дублей: {before} / {len(df)}")
print(df.groupby("symbol")["date"].agg(["min", "max", "count"]))