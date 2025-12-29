import os
import faiss
import torch
import numpy as np
from src.clip_embedder import CLIPEmbedder


class FaissImageIndex:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.embedder = CLIPEmbedder()
        self.image_paths = []
        self.index = None
        self._build_index()

    def _build_index(self):
        valid_ext = (".jpg", ".jpeg", ".png")

        image_paths = [
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if f.lower().endswith(valid_ext)
        ]

        print(f"FAISS: indexing {len(image_paths)} images...")

        embeddings = []
        for path in image_paths:
            emb = self.embedder.embed(path)
            embeddings.append(emb.cpu().numpy())
            self.image_paths.append(path)

        vectors = np.vstack(embeddings).astype("float32")
        dim = vectors.shape[1]

        self.index = faiss.IndexFlatIP(dim)  # cosine similarity via inner product
        self.index.add(vectors)

        print("FAISS index built.")