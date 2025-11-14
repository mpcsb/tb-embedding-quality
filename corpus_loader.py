
# corpus_loader.py
from __future__ import annotations
import re
import csv
from pathlib import Path
from typing import List, Tuple
import pandas as pd

# ---------------------------
# Generic text chunking utils
# ---------------------------
def _chunk_text(text: str, max_tokens=200, stride=80, min_chars=100) -> List[str]:
    toks = text.split()
    if not toks:
        return []
    out, i = [], 0
    while i < len(toks):
        w = " ".join(toks[i:i+max_tokens]).strip()
        if len(w) >= min_chars:
            out.append(w)
        if i + max_tokens >= len(toks):
            break
        i += max(1, max_tokens - stride)
    return out

def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen, out = set(), []
    for x in items:
        k = x.strip()
        if k and k not in seen:
            seen.add(k); out.append(k)
    return out

# ---------------------------
# Food corpus (single CSV)
# ---------------------------
def _read_csv_loose(path: Path) -> pd.DataFrame:
    """
    Robust CSV read: try default, then fallbacks with sep inference.
    """
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        # try python engine with sniffed delimiter
        with path.open("r", encoding="utf-8", newline="") as f:
            sample = "".join(f.readlines(5000))
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return pd.read_csv(path, low_memory=False, engine="python", sep=dialect.delimiter)

def load_food_chunks(
    csv_path: str | Path,
    target_chunks: int = 1000,
    max_tokens: int = 180,
    stride: int = 60,
    min_chars: int = 80,
) -> List[str]:
    """
    Build ~target_chunks from the Food Composition CSV by templating text fields.
    """
    path = Path(csv_path)
    df = _read_csv_loose(path)

    cols = {
        "name": "Food Name",
        "desc": "Food Description",
        "der":  "Derivation",
        "class":"Classification Name",
        "samp": "Sampling Details",
    }
    present = {k: c for k, c in cols.items() if c in df.columns}
    if not present:
        # fallback: any object columns
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        if not obj_cols:
            return []
        present = {f"c{i}": c for i, c in enumerate(obj_cols)}

    chunks: List[str] = []
    for _, r in df.iterrows():
        parts = []
        # structured template if the expected columns exist
        if "name" in present and isinstance(r.get(present["name"]), str):
            parts.append(f"Recipe ingredient: {r[present['name']]}.")
        if "desc" in present and isinstance(r.get(present["desc"]), str):
            parts.append(f"Description: {r[present['desc']]}.")
        if "der" in present and isinstance(r.get(present["der"]), str):
            parts.append(f"Derivation: {r[present['der']]}.")
        if "class" in present and isinstance(r.get(present["class"]), str):
            parts.append(f"Category: {r[present['class']]}.")
        if "samp" in present and isinstance(r.get(present["samp"]), str):
            parts.append(f"Sampling details: {r[present['samp']]}.")

        # if template produced nothing, coalesce any string columns
        if not parts:
            vals = [str(r[c]).strip() for c in present.values()
                    if isinstance(r.get(c), str) and r.get(c).strip()]
            para = " | ".join(vals)
        else:
            para = " ".join(p.strip() for p in parts if p and p.strip())

        if not para:
            continue

        chunks.extend(_chunk_text(para, max_tokens=max_tokens, stride=stride, min_chars=min_chars))
        if len(chunks) >= target_chunks:
            break

    return _dedupe_keep_order(chunks[:target_chunks])

# ---------------------------
# PubMed RCT (dir with .txt OR CSV fallback)
# ---------------------------
def _parse_rct_txt(path: Path) -> List[str]:
    """
    Parse one RCT .txt file:
      '###<id>' lines separate abstracts
      other lines: 'LABEL<TAB>SENTENCE'
    Returns: list of abstracts (each a single concatenated string).
    """
    abstracts, cur = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("###"):
                if cur:
                    abstracts.append(" ".join(cur))
                    cur = []
                continue
            parts = line.split("\t")
            sent = parts[-1].strip() if parts else ""
            if sent:
                cur.append(sent)
    if cur:
        abstracts.append(" ".join(cur))
    return abstracts

def _collect_rct_abstracts(dir_path: Path) -> List[str]:
    files = [dir_path / "train.txt", dir_path / "dev.txt", dir_path / "test.txt"]
    files = [p for p in files if p.exists()]
    if not files:
        files = list(dir_path.glob("*.txt"))
    abstracts: List[str] = []
    for p in files:
        abstracts.extend(_parse_rct_txt(p))
    return abstracts

def load_pubmed_chunks(
    dir_or_csv: str | Path,
    target_chunks: int = 1000,
    use_txt_first: bool = True,
    max_tokens: int = 220,
    stride: int = 90,
    min_chars: int = 120,
) -> List[str]:
    """
    Build ~target_chunks from PubMed RCT data:
      - If a directory: parse RCT .txt files (preferred: richer text).
      - Else: treat as CSV with columns [abstract_id, line_number, abstract_text],
              rebuild abstracts by grouping + concatenation.
    """
    p = Path(dir_or_csv)

    chunks: List[str] = []
    if use_txt_first and p.is_dir():
        abstracts = _collect_rct_abstracts(p)
        for abs_txt in abstracts:
            abs_txt = re.sub(r"\s+", " ", abs_txt).strip()
            chunks.extend(_chunk_text(abs_txt, max_tokens=max_tokens, stride=stride, min_chars=min_chars))
            if len(chunks) >= target_chunks:
                break
        return _dedupe_keep_order(chunks[:target_chunks])

    # CSV fallback (robust read)
    df = _read_csv_loose(p)
    required = {"abstract_id", "line_number", "abstract_text"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV missing required columns {required}")

    grouped = (
        df.sort_values(["abstract_id", "line_number"])
          .groupby("abstract_id")["abstract_text"]
          .apply(lambda s: " ".join(s.astype(str)))
    )
    for abs_txt in grouped:
        abs_txt = re.sub(r"\s+", " ", abs_txt).strip()
        chunks.extend(_chunk_text(abs_txt, max_tokens=max_tokens, stride=stride, min_chars=min_chars))
        if len(chunks) >= target_chunks:
            break
    return _dedupe_keep_order(chunks[:target_chunks])

# ---------------------------
# Public API
# ---------------------------
def load_corpora(
    food_csv: str | Path,
    pubmed_path: str | Path,  # dir with RCT .txt OR CSV file
    n_food: int = 1000,
    n_med: int = 1000,
) -> Tuple[List[str], List[str]]:
    food = load_food_chunks(food_csv, target_chunks=n_food)
    med  = load_pubmed_chunks(pubmed_path, target_chunks=n_med)
    return food, med
