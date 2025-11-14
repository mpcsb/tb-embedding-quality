# local_model_io.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer


def load_local_model(model_dir: str, device: Optional[str] = None) -> SentenceTransformer:
    """
    Load a SentenceTransformer from a local directory, offline-only.
    """
    # force offline behaviour once models are cached
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    p = Path(model_dir)
    if not p.exists():
        raise FileNotFoundError(f"Model dir not found: {p}")
    if not any(p.iterdir()):
        raise FileNotFoundError(f"Model dir exists but is empty: {p}")

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return SentenceTransformer(str(p), device=dev)


def ensure_local_model(model_name: str, local_dir: str, device: Optional[str] = None) -> SentenceTransformer:
    """
    Ensure `local_dir` contains a saved SentenceTransformer for `model_name`.

    - If `local_dir` exists and is non-empty: load offline from there.
    - Else: download `model_name`, save to `local_dir`, then reload offline.
    """
    p = Path(local_dir)

    # If already cached, just load offline
    if p.exists() and any(p.iterdir()):
        return load_local_model(str(p), device=device)

    # Need to download
    p.mkdir(parents=True, exist_ok=True)

    # make sure we're allowed to hit the hub
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⏳ Downloading {model_name} to {p} …")
    model = SentenceTransformer(model_name, device=dev)
    model.save(str(p))

    # now reload via the offline loader so subsequent calls are offline-safe
    return load_local_model(str(p), device=device)
