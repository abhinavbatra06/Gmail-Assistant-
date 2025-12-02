# app.py
import streamlit as st
import faiss, json, numpy as np, os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed(text, model="text-embedding-3-small"):
    resp = client.embeddings.create(model=model, input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

def retrieve(query, k=5):
    index = faiss.read_index("emails.index")
    metadata = json.load(open("metadata.json" ,  encoding="utf-8"))
    qv = embed(query)
    D, I = index.search(np.array([qv]), k)
    return [metadata[i] for i in I[0]]

def generate_answer(query, snippets, model="gpt-4o-mini"):
    context = "\n\n".join(
        [f"From: {s['from']}\nSubject: {s['subject']}\nDate: {s['date']}\n{s['text']}" for s in snippets]
    )
    prompt = f"""Answer strictly using the following email excerpts.
If the answer is not present, reply "Not found in emails."

Context:
{context}

Question: {query}
Answer:"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":"Answer only from provided emails."},
                  {"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content

# UI
st.title("📧 Email QA Prototype")
query = st.text_input("Ask a question about your emails:")

if query:
    snippets = retrieve(query, k=5)
    ans = generate_answer(query, snippets)
    st.write("### 💡 Answer")
    st.write(ans)
    with st.expander("📎 Sources"):
        for s in snippets:
            st.markdown(f"**{s['subject']}** — {s['date']} — {s['from']}")

