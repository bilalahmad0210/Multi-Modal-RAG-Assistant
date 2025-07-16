# 🧠 Multi-Modal RAG Assistant (Fully Local, LLM + OCR)

This is a **fully offline AI assistant** that can analyze and reason over:
- 📄 PDFs
- 🖼️ Images (with OCR text extraction)
- 🎥 Videos / Audio (via transcription)
- 💬 Natural language questions

All powered by local models (no internet or API keys needed).

---

## 🚀 Features

| Modality | What it can do |
|----------|----------------|
| 📄 PDF | Answer questions from uploaded PDFs or preloaded documents |
| 🖼️ Image | Extract image captions and visible text (OCR) |
| 🎙️ Audio / Video | Transcribe and answer questions about content |
| 🤖 Local LLM | Uses Mistral 7B (GGUF) via `llama-cpp-python` |
| 🔒 Offline | 100% privacy-safe and free. No API keys required |

---

## 🧰 Tech Stack

- **Gradio** – for UI
- **LangChain + FAISS** – for text & video retrieval
- **SentenceTransformers** – for vector embeddings
- **Mistral-7B Instruct (GGUF)** – for smart answers via `llama-cpp-python`
- **Tesseract OCR** – to read text from images (e.g., tweets, screenshots)
- **BLIP** – for image captioning
- **Whisper** – for video/audio transcription

---

## 📦 Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-username/multi-modal-rag-assistant.git
cd multi-modal-rag-assistant
```

### 2. Set up environment
```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

### 3. Download models

#### 🔡 Sentence Embedding
```bash
# Automatically downloaded: all-MiniLM-L6-v2
```

#### 🧠 LLM (Mistral 7B GGUF)
Download from: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF  
Put the `.gguf` file in:
```
models/mistral-7b-instruct.Q4_K_M.gguf
```

#### 🎙️ Whisper (Transcription)
```bash
pip install git+https://github.com/openai/whisper.git
```

#### 🔠 OCR (Tesseract)
- Install from: https://github.com/UB-Mannheim/tesseract/wiki
- Add install path (e.g., `C:\Program Files\Tesseract-OCR`) to System Environment Variables

---

## ✅ How to Run

```bash
python app.py
```

Visit: http://localhost:7860  
Ask a question and upload a file!

---

## 🧪 Example Prompts

| File | Prompt |
|------|--------|
| Resume.pdf | "What are this candidate's top skills?" |
| Tweet image | "What does the tweet say?" |
| Job PDF | "Summarize the job requirements." |
| Audio.mp3 | "What is this person talking about?" |

---

## 📁 Folder Structure

```
multi_modal_assistant/
├── app.py
├── llm.py
├── models/
│   └── mistral-7b-instruct.Q4_K_M.gguf
├── retriever/
│   ├── text.py
│   ├── image_utils.py
│   └── video.py
├── data/
│   ├── images/
│   ├── videos/
│   └── pdfs/
└── README.md
```

---

## ⚠️ Limitations

- High RAM (14GB+) recommended for local LLMs
- First inference may take a few seconds
- OCR accuracy depends on image quality
- FAISS index is in-memory and non-persistent by default

---

## 🧠 Roadmap

- [ ] Streamed LLM responses
- [ ] Chat memory and history
- [ ] Agent-based tool use (LangGraph)
- [ ] Upload multiple files
- [ ] Desktop or web API version

---

## 🧑‍💻 Author

**Bilal Ahmad** – AI Engineer, Builder, Learner  
> Powered by Mistral, LangChain, BLIP, Whisper, Tesseract, and Python 🐍

---

## 🛡️ License

MIT – use this assistant freely, locally, and privately.
