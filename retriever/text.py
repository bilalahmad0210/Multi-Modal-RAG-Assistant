# retriever/text.py

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

# GLOBALS
DB_PATH = "text_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)


def load_documents_from_folder(folder_path: str):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(folder_path, filename))
            docs.extend(loader.load())
    return docs


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)



def create_or_load_faiss_index(documents=None):
    index_file = os.path.join(DB_PATH, "index.faiss")
    meta_file = os.path.join(DB_PATH, "metadata.pkl")

    if os.path.exists(index_file) and os.path.exists(meta_file):
        print("🔁 Loading FAISS index and metadata...")
        index = faiss.read_index(index_file)
        with open(meta_file, "rb") as f:
            texts, metadata = pickle.load(f)
        return index, texts, metadata

    if documents is None:
        raise ValueError("❌ No documents provided and no index found.")

    print("⚙️ Creating FAISS index...")
    texts = [doc.page_content for doc in documents]
    metadata = [doc.metadata for doc in documents]
    embeddings = embedder.encode(texts, show_progress_bar=True)

    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index and metadata
    os.makedirs(DB_PATH, exist_ok=True)
    faiss.write_index(index, index_file)
    with open(meta_file, "wb") as f:
        pickle.dump((texts, metadata), f)

    return index, texts, metadata



def search(query, index_data, k=3):
    index, texts, metadata = index_data
    query_vec = embedder.encode([query])
    D, I = index.search(query_vec, k)
    return [texts[i] for i in I[0]]
