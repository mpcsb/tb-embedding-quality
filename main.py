
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import product
import pandas as pd
from tqdm.auto import tqdm
from z3 import Solver, Bool, RealVal, Or, Implies, sat 

# ---- local modules ----
from corpus_loader import load_corpora
from local_model_io import ensure_local_model      
from embed_utils import embed_texts          
from embed_utils import pca_project, quantize_uniform

# ==============================================================
# 1) LOAD CORPUS
# ==============================================================
FOOD = "data/Food Composition.csv"  #https://www.kaggle.com/datasets/vinitshah0110/food-composition
MED  = "data/PubMed_20k_RCT"       #https://www.kaggle.com/datasets/matthewjansen/pubmed-200k-rtc
food_chunks, medical_chunks = load_corpora(FOOD, MED, n_food=1000, n_med=1000)


# ==============================================================
# 2) LOAD MODELS FROM LOCAL DISK (NO DOWNLOAD)
# ==============================================================
# either the models exist here, or they get downloaded once and cached locally
# replace these models with whatever models you want to test
PATH_A = "./models_local/distilbert-base-nli-stsb-mean-tokens"
PATH_B = "./models_local/all-MiniLM-L6-v2"

mA = ensure_local_model("sentence-transformers/distilbert-base-nli-stsb-mean-tokens", PATH_A)  # 768d
mB = ensure_local_model("sentence-transformers/all-MiniLM-L6-v2", PATH_B)                      # 384d



# ==============================================================
# 3) EMBEDDINGS IN MEMORY (E[(corpus, model_variant)])
# ==============================================================

def degraded(X):
    Z64, _, _ = pca_project(X, out_dim=64)
    Z64 /= (np.linalg.norm(Z64, axis=1, keepdims=True) + 1e-12)
    Zq = quantize_uniform(Z64, bits=4, per_dim=True)
    Zq /= (np.linalg.norm(Zq, axis=1, keepdims=True) + 1e-12)
    return Zq.astype(np.float32)

E = {}

for corpus_name, chunks in [("food", food_chunks), ("medical", medical_chunks)]:
    print(f"\n🔹 embedding {corpus_name} (model A)")
    E[(corpus_name, "A_raw")] = embed_texts(mA, chunks, batch_size=256, normalize=True, progress=True)

    print(f"🔹 embedding {corpus_name} (model B)")
    E[(corpus_name, "B_raw")] = embed_texts(mB, chunks, batch_size=256, normalize=True, progress=True)

    print(f"🔹 PCA→64 + quant(4bit) {corpus_name} (model B_pca64q4)")
    E[(corpus_name, "B_pca64q4")] = degraded(E[(corpus_name, "B_raw")]) 


# ==============================================================
# 4) GEOMETRIC CONSISTENCY CHECK (Z3)
# ==============================================================

def cosine_distance_matrix_from_unit(X):
    S = X @ X.T
    np.clip(S, -1, 1, out=S)
    return (1.0 - S).astype(np.float32)

def knn_indices(D, k):
    idx = np.argsort(D, axis=1)
    return idx[:, 1:k+1].astype(np.int32)

def z3_violation(D, N, anchor, tau):
    i = int(anchor)
    s = Solver()
    bools = []
    for j in N[i]:
        j = int(j)
        for k in N[j]:
            if k == i or k == j: continue
            b = Bool(f"b_{i}_{j}_{k}")
            s.add(Implies(b, RealVal(float(D[i,j] + D[j,k])) < RealVal(float(D[i,k] - tau))))
            bools.append(b)
    if not bools:
        return False, None
    s.add(Or(*bools))
    out = s.check()
    if out == sat:
        m = s.model()
        for b in bools:
            if m.evaluate(b, model_completion=True):
                parts = b.decl().name().split("_")  # <-- CORRECT
                _, i, j, k = parts
                return True, (int(j), int(k))

    return False, None


def assess_local(X, k, tau, max_examples=5, desc=""):
    """Returns (summary_dict, examples)."""
    D = cosine_distance_matrix_from_unit(X)
    N = knn_indices(D, k)
    clean = 0
    examples = []
    for i in tqdm(range(X.shape[0]), desc=desc):
        sat, pair = z3_violation(D, N, i, tau)
        if not sat:
            clean += 1
        elif len(examples) < max_examples:
            j, k2 = pair
            examples.append((i, j, k2, float(D[i,j] + D[j,k2]), float(D[i,k2])))
    return clean, examples


# ==============================================================
# 5) SWEEP ALL MODELS & WRITE CSVs
# ==============================================================


stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

configs = [
    ("food",   "A_raw"),
    ("food",   "B_raw"),
    ("food",   "B_pca64q4"),
    ("medical","A_raw"),
    ("medical","B_raw"),
    ("medical","B_pca64q4"),
]

ks   = [2, 5] # higher K get's a bit more expensive, but worth exploring
taus = [0.0, 0.01, ] # tolerance margin

rows_metrics = []
rows_examples = []

for (corp, model), k, tau in product(configs, ks, taus):
    X = E[(corp, model)]
    desc = f"{corp}/{model} k={k} tau={tau}"
    clean, exs = assess_local(X, k=k, tau=tau, max_examples=6, desc=desc)

    rows_metrics.append({
        "timestamp": stamp,
        "corpus": corp,
        "model": model,
        "k": k,
        "tau": tau,
        "n": X.shape[0],
        "clean": clean,
        "clean_frac": clean / X.shape[0],
    })

    for (i, j, k2, lhs, rhs) in exs:
        rows_examples.append({
            "timestamp": stamp,
            "corpus": corp,
            "model": model,
            "k": k,
            "tau": tau,
            "i": i,
            "j": j,
            "k2": k2,
            "lhs": lhs,
            "rhs": rhs,
            "margin": rhs - lhs,
        })

# ---- write to disk ----
Path("./outputs").mkdir(exist_ok=True)

metrics_csv  = f"./outputs/geometry_metrics_{stamp}.csv"
examples_csv = f"./outputs/geometry_examples_{stamp}.csv"

pd.DataFrame(rows_metrics).to_csv(metrics_csv, index=False)
pd.DataFrame(rows_examples).to_csv(examples_csv, index=False)

print("\n saved results:")
print(f"  metrics  --> {metrics_csv}")
print(f"  examples --> {examples_csv}")