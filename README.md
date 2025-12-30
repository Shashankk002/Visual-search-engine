Visual Image Search Engine (CLIP + FAISS)

This project implements a scalable visual image search engine that retrieves the top-K most visually similar images for a given query image. The system uses CLIP embeddings for semantic representation and FAISS for efficient approximate nearest-neighbor (ANN) search.

The goal of this project is not to build a toy demo, but to explore real system behavior, performance trade-offs, and failure modes that arise in practical visual search applications.

⸻

Overview

Given a query image, the system:
	1.	Encodes the image using a pretrained CLIP vision model
	2.	Projects it into a normalized embedding space
	3.	Searches a FAISS index to retrieve the most similar images
	4.	Returns ranked results based on cosine similarity

The system treats visual search as a ranking problem, not a binary decision task.

⸻

System Architecture

Query Image
    ↓
CLIP Image Encoder (ViT-B/32)
    ↓
512-D Normalized Embedding
    ↓
FAISS ANN Index (Inner Product)
    ↓
Top-K Similar Images (Ranked)

Key design choice:
	•	Similarity is reported as a score and used for ranking, not threshold-based filtering.

⸻

Project Structure

visual_search/
├── app.py                 # Streamlit demo UI
├── core/
│   └── search_engine.py   # Shared visual search logic
├── src/
│   ├── clip_embedder.py   # CLIP embedding module
│   ├── faiss_index.py    # FAISS index builder
│   └── faiss_search.py   # FAISS search logic
├── data/
│   └── images/            # Dataset (e.g. Airbnb images)
└── README.md

The core search logic is independent of the UI and can be reused in different interfaces or services.

⸻

Dataset

The system was evaluated using multiple real-world datasets, including:
	•	Airbnb Duplicate Image Detection dataset, focusing on room-level categories such as:
	•	Living rooms
	•	Bedrooms
	•	Bathrooms
	•	Stanford Online Products (SOP) dataset, which contains product images with high intra-class similarity and large inter-class variation

These datasets are challenging because:
	•	Each entity (listing or product) contains multiple images
	•	Different entities often look visually similar
	•	Images vary significantly in lighting, angle, background, and composition

This makes instance-level retrieval non-trivial and realistic.

⸻

Example Behavior

When querying a living-room image, typical results include:
	1.	The exact same image (similarity close to 1.0)
	2.	Images from the same listing taken from different angles
	3.	Images from the same listing under different lighting
	4.	Visually similar rooms from other listings

This behavior is expected for semantic visual search.

⸻

Visual Search vs Deduplication

This project intentionally focuses on visual similarity search, not strict duplicate detection.
	•	Visual search answers: “Which images look most similar?”
	•	Deduplication answers: “Which images belong to the same physical object or listing?”

CLIP embeddings capture semantic and scene-level similarity, so visually similar rooms across different listings may rank highly even if they are not duplicates.

Solving deduplication requires additional constraints (such as metadata or geometric verification) and is handled separately.

⸻

Performance

Example run on approximately 190 images (CPU-only):

Stage	Time
Index Build	~7.3 s
Search	~0.03 s
Total	~9.4 s

Notes:
	•	Index build time is dominated by CLIP forward passes
	•	Query latency scales efficiently due to FAISS
	•	Brute-force search was used initially for validation, then removed

⸻

Evaluation Philosophy

Instead of relying on arbitrary similarity thresholds, the system is evaluated using:
	•	Top-K retrieval quality
	•	Same-listing recall in the top-K results
	•	Qualitative analysis of failure cases

This mirrors how visual search systems are evaluated in practice.

⸻

Technology Stack
	•	Python
	•	PyTorch
	•	CLIP
	•	FAISS (CPU)
	•	Streamlit (demo UI)

⸻

Key Takeaways
	•	Semantic embeddings capture scene similarity, not instance identity
	•	Ranking-based retrieval is more appropriate than thresholding
	•	Approximate nearest-neighbor search is essential for scalability
	•	Real datasets expose meaningful failure modes
	•	Correct problem framing is as important as model choice

⸻

Future Work
	•	Metadata-aware filtering (e.g. listing-level constraints)
	•	Batch querying and caching
	•	Recall vs latency benchmarking at larger scale
	•	Backend API for deployment
	•	Cross-modal search (text-to-image)

⸻

Conclusion

This project demonstrates an end-to-end visual image search system with realistic behavior, measured performance, and clearly understood limitations.

The emphasis is on engineering correctness, interpretability, and system-level reasoning, rather than optimizing for a single metric or producing a superficial demo.
---
Note: Dataset files are excluded. 
Provide your own image directory when running the application.
