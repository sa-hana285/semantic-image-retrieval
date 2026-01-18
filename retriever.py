import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class ImageRetriever:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path).dropna(subset=["description", "filename"])
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.embeddings = self.model.encode(
            self.df["description"].tolist(),
            show_progress_bar=True
        ).astype("float32")

        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query, k=3):
        query_vec = self.model.encode([query]).astype("float32")
        D, I = self.index.search(query_vec, k)

        results = []
        for idx, dist in zip(I[0], D[0]):
            row = self.df.iloc[idx]
            results.append({
                "filename": row["filename"],
                "description": row["description"],
                "score": dist
            })
        return results
