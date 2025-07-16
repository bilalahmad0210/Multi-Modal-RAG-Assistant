# retriever/video.py

import os
import whisper
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Globals
DB_PATH = "video_index"
TRANSCRIPT_PATH = "data/videos/"
embedder = SentenceTransformer("all-MiniLM-L6-v2")
whisper_model = whisper.load_model("base")


def transcribe_video(video_path):
    print(f"🎙️ Transcribing {video_path}...")
    result = whisper_model.transcribe(video_path)
    return result["text"]


def split_and_embed(text, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(text)
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    return chunks, embeddings


def create_or_load_video_index():
    index_file = os.path.join(DB_PATH, "index.faiss")
    meta_file = os.path.join(DB_PATH, "metadata.pkl")

    if os.path.exists(index_file) and os.path.exists(meta_file):
        print("🔁 Loading FAISS video index...")
        index = faiss.read_index(index_file)
        with open(meta_file, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks

    print("⚙️ Creating FAISS video index...")
    full_text = ""

    for file in os.listdir(TRANSCRIPT_PATH):
        if file.endswith((".mp4", ".mp3", ".wav", ".m4a")):
            path = os.path.join(TRANSCRIPT_PATH, file)
            full_text += transcribe_video(path) + "\n"

    if not full_text.strip():
        raise ValueError("❌ No audio or video transcribed.")

    chunks, embeddings = split_and_embed(full_text)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(DB_PATH, exist_ok=True)
    faiss.write_index(index, index_file)
    with open(meta_file, "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks


def search_video(query, index_data, k=3):
    index, chunks = index_data
    query_vec = embedder.encode([query])
    D, I = index.search(query_vec, k)
    return [chunks[i] for i in I[0]]
