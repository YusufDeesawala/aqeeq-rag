# app.py
import os
import json
import time
import sqlite3
import numpy as np
import faiss
import requests
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# ---------- CONFIG ----------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

SQLITE_PATH = os.path.join(DATA_DIR, "qa_store.db")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH = os.path.join(DATA_DIR, "meta.json")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_REST_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

EMBED_DIM = 384         # MiniLM output
REDUCED_DIM = 64        # target dim
PCA_THRESHOLD = 50      # train PCA after this many samples

# ---------- INIT ----------
app = Flask(__name__)

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# SQLite
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

# PCA + Index state
pca = None
use_reduced = False
if os.path.exists(META_PATH):
    with open(META_PATH, "r") as f:
        meta = json.load(f)
        use_reduced = meta.get("use_reduced", False)

if use_reduced:
    index = faiss.IndexFlatIP(REDUCED_DIM)
else:
    index = faiss.IndexFlatIP(EMBED_DIM)

# Load stored vectors if exist
if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)

# ---------- HELPERS ----------
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
    X = embed_model.encode(texts, convert_to_numpy=True).astype("float32")
    pca = PCA(n_components=REDUCED_DIM)
    X_reduced = pca.fit_transform(X)
    X_reduced = normalize(X_reduced)
    index = faiss.IndexFlatIP(REDUCED_DIM)
    index.add(X_reduced)
    use_reduced = True
    save_state()
    print("✅ PCA applied, index rebuilt.")

def embed_text(text: str) -> np.ndarray:
    v = embed_model.encode([text], convert_to_numpy=True).astype("float32")
    if use_reduced and pca is not None:
        v = pca.transform(v)
    v = normalize(v)
    return v

def query_gemini(prompt: str) -> str:
    url = f"{GEMINI_REST_BASE}/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code != 200:
        return f"Error from Gemini: {resp.text}"
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]

# ---------- ROUTES ----------
@app.route("/add", methods=["POST"])
def add():
    global use_reduced
    data = request.get_json()
    q, a = data.get("question"), data.get("answer")
    if not q or not a:
        return jsonify({"error": "need question and answer"}), 400

    # Insert into DB
    cur = db_conn.cursor()
    cur.execute("INSERT INTO qa (question, answer, created_at) VALUES (?, ?, ?)",
                (q, a, time.time()))
    db_conn.commit()

    # Embedding
    emb = embed_text(q)
    index.add(emb)

    # Train PCA if threshold crossed
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

    emb = embed_text(query)
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
    prompt = f"Answer the following user query using the provided context.\n\nContext:\n{context}\n\nUser Query: {query}"

    answer = query_gemini(prompt)
    return jsonify({"answer": answer, "retrieved": retrieved})

# ---------- MAIN ----------
if __name__ == "__main__":
    app.run(debug=True)
