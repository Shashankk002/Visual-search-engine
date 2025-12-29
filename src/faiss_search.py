import numpy as np


def faiss_search(query_embedding, faiss_index, image_paths, top_k=5):
    query = query_embedding.cpu().numpy().astype("float32").reshape(1, -1)

    scores, indices = faiss_index.search(query, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append((image_paths[idx], float(score)))

    return results