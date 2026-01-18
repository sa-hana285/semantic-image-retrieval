import streamlit as st
from PIL import Image
from retriever import ImageRetriever
import os

st.set_page_config(
    page_title="Semantic Image Retrieval",
    layout="centered"
)

st.title("Semantic Image Retrieval System")
st.write("Search images using natural language (dataset-limited)")

@st.cache_resource
def load_retriever():
    return ImageRetriever("data1.csv")

retriever = load_retriever()

query = st.text_input("Enter your search query")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if query:
    results = retriever.search(query, k=3)

    st.subheader("Results")
    cols = st.columns(3)

    for col, res in zip(cols, results):
        try:

            img_path = os.path.join(BASE_DIR, res["filename"])
            img = Image.open(img_path)

            col.image(img, width=300)
            col.caption(f"Score: {res['score']:.4f}")

        except Exception as e:
            col.error(f"Image not found: {e}")
