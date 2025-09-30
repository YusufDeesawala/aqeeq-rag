import os
import json
import time
import sqlite3
import numpy as np
import faiss
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import requests

# ---------- CONFIG ----------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

SQLITE_PATH = os.path.join(DATA_DIR, "qa_store.db")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH = os.path.join(DATA_DIR, "meta.json")

REDUCED_DIM = 64
PCA_THRESHOLD = 50

# Groq Config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ---------- INIT ----------
app = Flask(__name__)

def init_db():
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS qa (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at REAL NOT NULL
                   )""")
    conn.commit()
    return conn

db_conn = init_db()

# PCA and FAISS
pca = None
use_reduced = False
embed_dim = 384

if os.path.exists(META_PATH):
    with open(META_PATH, "r") as f:
        meta = json.load(f)
        use_reduced = meta.get("use_reduced", False)

index = faiss.IndexFlatIP(REDUCED_DIM if use_reduced else embed_dim)

if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)

# ---------- MiniLM Embedding ----------
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def embed_text(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        token_emb = outputs.last_hidden_state
        attention_mask = inputs.attention_mask.unsqueeze(-1)
        masked = token_emb * attention_mask
        summed = masked.sum(dim=1)
        counts = attention_mask.sum(dim=1)
        mean_emb = summed / torch.clamp(counts, min=1e-9)
        emb = mean_emb.squeeze().cpu().numpy()
    return emb.astype("float32")

# ---------- Helpers ----------
def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

def save_state():
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(META_PATH, "w") as f:
        json.dump({"use_reduced": use_reduced}, f)

def get_all_texts():
    cur = db_conn.cursor()
    cur.execute("SELECT question FROM qa")
    return [row[0] for row in cur.fetchall()]

def rebuild_with_pca():
    global index, pca, use_reduced
    print("🔄 Rebuilding FAISS index with PCA...")
    texts = get_all_texts()
    if not texts:
        return
    X = np.vstack([embed_text(t) for t in texts]).astype("float32")
    pca = PCA(n_components=REDUCED_DIM)
    X_reduced = pca.fit_transform(X)
    X_reduced = normalize(X_reduced)
    index = faiss.IndexFlatIP(REDUCED_DIM)
    index.add(X_reduced)
    use_reduced = True
    save_state()
    print("✅ PCA applied, index rebuilt.")

# ---------- Groq Completion ----------
def query_groq(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    resp = requests.post(GROQ_API_URL, headers=headers, json=data)
    if resp.status_code != 200:
        raise RuntimeError(f"Groq API error: {resp.text}")
    response_data = resp.json()
    return response_data["choices"][0]["message"]["content"]

# ---------- Routes ----------
@app.route("/add", methods=["POST"])
def add():
    global use_reduced
    data = request.get_json()
    q = data.get("question")
    a = data.get("answer")
    if not q or not a:
        return jsonify({"error": "need question and answer"}), 400

    cur = db_conn.cursor()
    cur.execute("INSERT INTO qa (question, answer, created_at) VALUES (?, ?, ?)",
                (q, a, time.time()))
    db_conn.commit()

    emb = embed_text(q).reshape(1, -1)
    index.add(emb)

    cur.execute("SELECT COUNT(*) FROM qa")
    count = cur.fetchone()[0]

    if not use_reduced and count >= PCA_THRESHOLD:
        rebuild_with_pca()

    save_state()
    return jsonify({"status": "ok", "count": count})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query")
    k = int(data.get("top_k", 3))
    if not query:
        return jsonify({"error": "need query"}), 400

    emb = embed_text(query).reshape(1, -1)
    D, I = index.search(emb, k)

    cur = db_conn.cursor()
    cur.execute("SELECT id, question, answer FROM qa")
    rows = cur.fetchall()
    id_map = {i: row for i, row in enumerate(rows)}

    retrieved = []
    for idx in I[0]:
        if idx < 0 or idx >= len(rows):
            continue
        _, q, a = id_map[idx]
        retrieved.append({"question": q, "answer": a})

    context = "\n".join([f"Q: {r['question']}\nA: {r['answer']}" for r in retrieved])

    prompt = (
        "Use the following context to answer the user query.\n\n"
        f"Context:\n{context}\n\n"
        f"User Query: {query}\n\n"
        "If the context is insufficient, answer using your own knowledge."
    )

    answer = query_groq(prompt)
    return jsonify({"answer": answer, "retrieved": retrieved})

if __name__ == "__main__":
    app.run(debug=True, port=8000)
