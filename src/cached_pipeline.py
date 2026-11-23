import os
import json
import hashlib
import pandas as pd
from pipeline import run_pipeline, RESULT_COLUMNS

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _key(question, answer):
    raw = json.dumps({"q": question, "a": answer}, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def run_cached(question=None, answer=None, use_cache=True):
    """Same as run_pipeline, but saves results. Returns (title, answer, df, from_cache)."""
    path = os.path.join(CACHE_DIR, _key(question, answer) + ".json")
    if use_cache and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        rows = d.get("rows", [])
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=RESULT_COLUMNS)
        return d["title"], d["answer"], df, True

    title, ans, df = run_pipeline(question=question, answer=answer)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"title": title, "answer": ans, "rows": df.to_dict("records")},
            f,
            ensure_ascii=False,
            indent=2,
        )
    return title, ans, df, False
