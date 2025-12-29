import argparse
import time

from src.clip_embedder import CLIPEmbedder
from src.faiss_index import FaissImageIndex
from src.faiss_search import faiss_search


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Image Search Engine")

    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    # Build index
    t0 = time.time()
    index = FaissImageIndex(args.image_dir)
    t1 = time.time()

    # Embed query
    embedder = CLIPEmbedder()
    query_embedding = embedder.embed(args.query)

    # Search
    t2 = time.time()
    results = faiss_search(
        query_embedding,
        index.index,
        index.image_paths,
        top_k=args.top_k,
    )
    t3 = time.time()

    print("\nTop results:")
    for path, score in results:
        print(f"{path}  (similarity={score:.3f})")

    print("\nTiming:")
    print(f"Index build: {t1 - t0:.4f}s")
    print(f"Search:      {t3 - t2:.6f}s")
    print(f"Total:       {t3 - t0:.4f}s")