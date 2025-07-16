# llm.py

from llama_cpp import Llama
import os

MODEL_PATH = "models/mistral-7b-instruct-v0.1.Q4_K_S.gguf"  # adjust path if needed

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=os.cpu_count() // 2,
    n_gpu_layers=20,  # adjust if OOM
    use_mlock=True,
    verbose=False
)

def ask_local_llm(question, context, max_tokens=512):
    prompt = f"""[INST] Use the context below to answer the question clearly.

### Context:
{context}

### Question:
{question}
[/INST]"""

    output = llm(prompt, max_tokens=max_tokens, stop=["</s>"])
    return output["choices"][0]["text"].strip()
