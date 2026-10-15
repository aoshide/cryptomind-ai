"""Корпус новостей Lenta.ru с HuggingFace (ПР-6). Стриминг: качаем только нужное."""
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed" / "lenta_corpus.csv"

TOPICS = ["Экономика", "Спорт", "Наука и техника", "Культура"]
PER_TOPIC = 150                     # 4 x 150 = 600 >= 500

ds = load_dataset("zloelias/lenta-ru", split="train", streaming=True)
print("Колонки:", ds.column_names)

ds = ds.shuffle(buffer_size=20000, seed=42)   # локально перемешали поток

collected = {t: 0 for t in TOPICS}
rows = []
for ex in ds:
    t = ex["topic"]
    if t in TOPICS and collected[t] < PER_TOPIC:
        text = f'{ex.get("title") or ""}. {ex.get("text") or ""}'[:1800]
        rows.append({"text": text, "topic": t})
        collected[t] += 1
        if all(v >= PER_TOPIC for v in collected.values()):
            break

corpus = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
corpus.to_csv(OUT, index=False)
print({t: collected[t] for t in TOPICS})
print(f"Итого: {len(corpus)} документов -> {OUT}")