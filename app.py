import streamlit as st
import time
from PIL import Image

from src.clip_embedder import CLIPEmbedder
from src.faiss_index import FaissImageIndex
from src.faiss_search import faiss_search


st.set_page_config(page_title="Visual Image Search", layout="wide")

st.title("🔍 Visual Image Search Engine")
st.write("CLIP embeddings + FAISS ANN search")

# Sidebar
st.sidebar.header("Settings")
image_dir = st.sidebar.text_input(
    "Image directory",
    value="data/images/"
)
top_k = st.sidebar.slider("Top-K results", 1, 10, 5)

# Cache index so it builds only once
@st.cache_resource
def load_index(dir_path):
    return FaissImageIndex(dir_path)

index = load_index(image_dir)
embedder = CLIPEmbedder()

# Upload query image
uploaded_file = st.file_uploader(
    "Upload a query image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    query_image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Query Image")
    st.image(query_image, width=300)

    # Save temp file
    temp_path = "temp_query.jpg"
    query_image.save(temp_path)

    # Search
    t0 = time.time()
    query_embedding = embedder.embed(temp_path)
    results = faiss_search(
        query_embedding,
        index.index,
        index.image_paths,
        top_k=top_k
    )
    t1 = time.time()

    st.subheader("Top Results")

    cols = st.columns(top_k)
    for col, (path, score) in zip(cols, results):
        with col:
            st.image(path, width="stretch")
            st.caption(f"{score:.3f}")

    st.write(f"⏱ Search time: {t1 - t0:.4f} seconds")