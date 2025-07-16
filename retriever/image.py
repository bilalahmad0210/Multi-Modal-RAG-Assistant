# retriever/image.py

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
import faiss
import pickle
import pytesseract
from sentence_transformers import SentenceTransformer
# Globals
DB_PATH = "image_index"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def generate_caption(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def create_or_load_image_index(image_folder="data/images"):
    index_file = os.path.join(DB_PATH, "index.faiss")
    meta_file = os.path.join(DB_PATH, "metadata.pkl")

    if os.path.exists(index_file) and os.path.exists(meta_file):
        print("🔁 Loading image FAISS index...")
        index = faiss.read_index(index_file)
        with open(meta_file, "rb") as f:
            captions, image_paths = pickle.load(f)
        return index, captions, image_paths

    print("⚙️ Creating image FAISS index...")

    captions, image_paths = [], []
    for img_file in os.listdir(image_folder):
        if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(image_folder, img_file)
            caption = generate_caption(path)
            captions.append(caption)
            image_paths.append(path)

    # Embed captions
    embeddings = embedder.encode(captions, show_progress_bar=True)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(DB_PATH, exist_ok=True)
    faiss.write_index(index, index_file)
    with open(meta_file, "wb") as f:
        pickle.dump((captions, image_paths), f)

    return index, captions, image_paths


def search_images(query, index_data, k=3):
    index, captions, image_paths = index_data
    query_vec = embedder.encode([query])
    D, I = index.search(query_vec, k)
    results = [{"caption": captions[i], "image_path": image_paths[i]} for i in I[0]]
    return results



def extract_ocr_text(image_path_or_file):
    try:
        img = Image.open(image_path_or_file)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"❌ OCR failed: {str(e)}"
