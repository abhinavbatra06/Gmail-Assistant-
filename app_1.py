import streamlit as st
import faiss, json, numpy as np, os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- Retrieval ----
def embed(text, model="text-embedding-3-small"):
    resp = client.embeddings.create(model=model, input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

def retrieve(query, k=5):
    index = faiss.read_index("emails.index")
    with open("metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    qv = embed(query)
    D, I = index.search(np.array([qv]), k)
    return [metadata[i] for i in I[0]]

# ---- Generation ----
def generate_answer(chat_history, query, snippets, model="gpt-4o-mini"):
    context = "\n\n".join(
        [f"From: {s['from']}\nSubject: {s['subject']}\nDate: {s['date']}\n{s['text']}" for s in snippets]
    )

    system_prompt = "You are a helpful assistant that answers strictly based on provided email context." \
    "You are answering to students.Also try answering for events in the future with respect to today's date as much as possible If the answer is not present, reply: Not found in emails."

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)  # previous Q&A
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    })

    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content

# ---- Streamlit App ----
st.title("CDS assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask me about your emails..."):
    # Display user input
    with st.chat_message("user"):
        st.markdown(query)

    snippets = retrieve(query, k=5)
    answer = generate_answer(st.session_state.chat_history, query, snippets)

    # Save to history
    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)

    # (Optional) Show sources
    with st.expander("📎 Sources"):
        for s in snippets:
            st.markdown(f"**{s['subject']}** — {s['date']} — {s['from']}")
