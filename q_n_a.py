

# query_cli.py
import os, json
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed(text, model="text-embedding-3-small"):
    resp = client.embeddings.create(model=model, input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

def retrieve(query, k=5):
    index = faiss.read_index("emails.index")
    metadata = json.load(open("metadata.json", "r", encoding="utf-8"))

    qv = embed(query)
    D, I = index.search(np.array([qv]), k)
    hits = [metadata[i] for i in I[0]]
    return hits

def generate_answer(query, snippets, model="gpt-4o-mini"):
    # Keep context compact
    context = ""
    for s in snippets:
        context += f"From: {s['from']}\nSubject: {s['subject']}\nDate: {s['date']}\nText: {s['text']}\n\n"

    prompt = f"""You answer strictly using the provided email excerpts. You are answering to students.Also try answering for events in the future as much as possible
If the answer is not present, reply: "Not found in emails."

Context:
{context}

Question: {query}
Answer:"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You are a precise assistant that cites only the provided email context."},
            {"role":"user","content": prompt}
        ]
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    print("🔎 Email QA ready. Type your questions (type 'exit' to quit).")
    while True:
        q = input("\n❓ Question: ").strip()
        if q.lower() in {"exit","quit"}:
            break
        hits = retrieve(q, k=5)
        ans = generate_answer(q, hits, model="gpt-4o-mini")
        print("\n💡 Answer:", ans)
        print("\n📎 Sources:")
        for h in hits:
            print(f"- {h['subject']} — {h['date']} — {h['from']}")
