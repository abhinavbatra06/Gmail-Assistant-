# build_index.py
import os, json, math
import numpy as np
import faiss, tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chunk_text(text, max_tokens=300, overlap=50):
    if not text: return []
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)
    if not toks: return []
    chunks = []
    step = max_tokens - overlap
    i = 0
    while i < len(toks):
        j = min(i + max_tokens, len(toks))
        chunks.append(enc.decode(toks[i:j]))
        i += step if step > 0 else max_tokens
    return chunks

def embed_batch(texts, model="text-embedding-3-small", batch_size=64):
    """Returns a list[np.ndarray] embeddings for the input texts."""
    out = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        for d in resp.data:
            out.append(np.array(d.embedding, dtype=np.float32))
    return out

if __name__ == "__main__":
    emails = json.load(open("emails.json", "r", encoding="utf-8"))
    print(f"Indexing {len(emails)} emails…")

    metadata = []
    all_chunks = []
    # Build chunks + metadata first (single pass)
    for e in emails:
        chunks = chunk_text(e.get("body", ""))
        for i, ch in enumerate(chunks):
            metadata.append({
                "id": f'{e["id"]}_chunk{i}',
                "from": e.get("from", ""),
                "subject": e.get("subject", ""),
                "date": e.get("date", ""),
                "text": ch
            })
            all_chunks.append(ch)

    if not all_chunks:
        raise SystemExit("No chunks to embed. Check emails.json content.")

    # Batch embed for speed
    vecs = embed_batch(all_chunks, model="text-embedding-3-small", batch_size=64)
    dim = len(vecs[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.vstack(vecs))
    faiss.write_index(index, "emails.index")

    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote FAISS index ({index.ntotal} vectors) and metadata.json")
