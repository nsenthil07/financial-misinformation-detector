"""Step 5 - Retrieval: the "R" in Retrieval-Augmented Generation (RAG).

An LLM on its own may "hallucinate" (confidently make things up). RAG
reduces this: first SEARCH a trusted knowledge base, then give the LLM the
best matches so it answers from evidence and can cite it.

This file is a tiny vector database:
  1. every knowledge-base document is turned into an embedding vector
  2. a query is embedded the same way
  3. cosine similarity (how closely two vectors point the same way) finds
     the closest documents

Two knowledge bases are used:
  * claims   - past professional fact-checks (TRAINING split only, so we never
               retrieve the answer to a test question)
  * messages - 10 known scam / legitimate patterns written from FTC guidance
               (data/knowledge_base/scam_patterns.json)

Real systems use a vector database such as Chroma, pgvector or Azure AI
Search. The idea (embed, then find nearest by cosine similarity) is the same.
"""
import json

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import make_embeddings


class VectorStore:
    """Holds documents (a DataFrame with columns text, label, evidence, url)
    and their embedding vectors."""

    def __init__(self, documents, embeddings="lsa", extra_texts=None):
        """`extra_texts`: more text used ONLY to learn the embedding vocabulary.
        A 10-document knowledge base is too small to learn word meanings from,
        so for the scam patterns we also learn from the training messages."""
        self.docs = documents.reset_index(drop=True)
        corpus = list(self.docs["text"] + " " + self.docs["evidence"]) + list(extra_texts or [])
        n_dims = min(config.EMBEDDING_DIM, len(corpus) - 1)
        self.embedder = make_embeddings(embeddings, n_dims).fit(corpus)
        self.vectors = self.embed(self.docs["text"])

    def embed(self, texts):
        vectors = np.asarray(self.embedder.transform(list(texts)), dtype=np.float32)
        lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(lengths, 1e-9)  # unit length -> dot product = cosine similarity

    def search(self, query, k=4):
        """Return the k most similar documents as a list of dicts."""
        similarities = self.vectors @ self.embed([query])[0]
        best = np.argsort(similarities)[::-1][:k]
        results = []
        for i in best:
            doc = self.docs.iloc[i]
            results.append({
                "text": doc["text"],
                "label": None if pd.isna(doc["label"]) else int(doc["label"]),
                "evidence": doc["evidence"],
                "url": doc["url"],
                "similarity": float(similarities[i]),
            })
        return results

    def knn_vote(self, queries, k=10):
        """Retrieval-only baseline: score = similarity-weighted share of the
        k nearest documents that were rated false/scam."""
        similarities = self.embed(queries) @ self.vectors.T
        labels = self.docs["label"].to_numpy(dtype=float)
        scores = []
        for row in similarities:
            best = np.argsort(row)[::-1][:k]
            weights = np.clip(row[best], 1e-6, None)
            scores.append(float((weights * labels[best]).sum() / weights.sum()))
        return np.array(scores)

    def save(self, path):
        joblib.dump(self, path, compress=3)


def load_scam_patterns(path=config.KNOWLEDGE_BASE / "scam_patterns.json"):
    """Read the scam-pattern JSON file into the same DataFrame format."""
    items = json.loads(open(path, encoding="utf-8").read())
    return pd.DataFrame({
        "text": [i["pattern"] + ". " + " ".join(i.get("examples", [])) for i in items],
        "label": [i.get("label") for i in items],
        "evidence": [i.get("advice", "") for i in items],
        "url": [i.get("url", "") for i in items],
    })
