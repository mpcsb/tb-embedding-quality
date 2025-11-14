# Embedding Geometry

This repository contains the exact code used to generate the figures and results for the blog post:

**https://www.testingbranch.com/embedding-quality/**

It is a **snapshot**, not a maintained library.  
All results are reproducible with the included environment and scripts.


## Environment

This project uses uv for environment management.

To set up:

```bash
uv sync
uv run main.py
```

What this does:

- loads chunked corpora from data/
- downloads the models once into models_local/
- computes embeddings (raw + PCA64 → 4-bit quantized)
- runs Z3 geometric consistency checks
- writes CSV results to outputs/

 