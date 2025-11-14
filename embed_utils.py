
# emb_utils.py
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def load_st_model(name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(name, device=device)
    return model

from tqdm.auto import tqdm
import math
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

@torch.no_grad()
def embed_texts(
    model: SentenceTransformer,
    texts,
    batch_size: int = 128,
    normalize: bool = False,
    progress: bool = True,
    desc: str = "Embedding",
):
    texts = list(texts)
    n = len(texts)
    out = []
    for i in tqdm(range(0, n, batch_size),
                  total=math.ceil(n / batch_size),
                  disable=not progress,
                  desc=desc):
        batch = texts[i:i+batch_size]
        vecs = model.encode(
            batch,
            batch_size=batch_size,           # internal batching still helps on tokenization
            convert_to_numpy=True,
            normalize_embeddings=False,      # we handle below
            show_progress_bar=False,         # we control tqdm
        )
        out.append(vecs)

    X = np.vstack(out).astype(np.float32)
    if normalize:
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X  # [N, D]


def pca_project(X: np.ndarray, out_dim: int):
    Xc = X - X.mean(axis=0, keepdims=True)
    # economy SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:out_dim].T  # [D, out_dim]
    Z = Xc @ W          # [N, out_dim]
    return Z, W, X.mean(axis=0, keepdims=True)

def quantize_uniform(X: np.ndarray, bits=4, per_dim=True):
    q = 2**bits - 1
    if per_dim:
        mn = X.min(axis=0, keepdims=True)
        mx = X.max(axis=0, keepdims=True)
    else:
        mn = np.array([[X.min()]])
        mx = np.array([[X.max()]])
    span = (mx - mn) + 1e-12
    Y = (X - mn) / span
    Yq = np.round(Y * q) / q
    Xq = Yq * span + mn
    return Xq