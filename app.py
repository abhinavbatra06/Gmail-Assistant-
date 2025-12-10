"""
Gmail Assistant - Streamlit Chat UI
Simple chat interface for querying emails
Integrated with NoticeBoard RAG system
"""
import streamlit as st
import time
import uuid
from src.rag_query import RAGQuery

# Page config
st.set_page_config(
    page_title="Gmail Assistant",
    page_icon="",
    layout="wide"
)

def display_sources(sources: list):
    """Display sources in a simple, readable format."""
    if not sources:
        st.info("No sources found.")
        return
    
    for i, source in enumerate(sources[:5], 1):
        subject = source.get('subject', 'No subject')
        sender = source.get('from', 'Unknown sender')
        date = source.get('date', 'Unknown date')
        source_type = source.get('source_type', 'unknown')
        
        with st.container(border=True):
            st.markdown(f"**Source {i}**")
            st.markdown(f"**Subject:** {subject}")
            st.markdown(f"**From:** {sender}")
            st.markdown(f"**Date:** {date}")
            st.markdown(f"**Type:** {source_type}")

# Title
st.title("Gmail Assistant")
st.caption("Ask questions about your emails")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Generate session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "metadata" in message:
            with st.expander("Details"):
                metadata = message["metadata"]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Intent", metadata.get("intent", "N/A"))
                with col2:
                    st.metric("Chunks Retrieved", metadata.get("num_chunks", 0))
                with col3:
                    st.metric("Response Time", f"{metadata.get('latency_ms', 0)/1000:.1f}s")
                
                if metadata.get("sources"):
                    st.markdown("### Sources")
                    display_sources(metadata["sources"])

# Chat input
if prompt := st.chat_input("Ask about your emails..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare chat history for RAG
    chat_history = (
        st.session_state.messages[-10:]
        if len(st.session_state.messages) > 10
        else st.session_state.messages
    )

    with st.chat_message("assistant"):
        with st.spinner("Searching emails..."):
            start_time = time.time()

            try:
                rag = RAGQuery()

                # Pass user query + chat history
                result = rag.query(
                    user_query=prompt,
                    chat_history=chat_history,
                    session_id=st.session_state.session_id
                )

                answer = result.get("answer", "Sorry, I couldn't find an answer.")
                st.markdown(answer)

                metadata = {
                    "intent": result.get("intent", "N/A"),
                    "num_chunks": result.get("num_chunks_retrieved", 0),
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "sources": result.get("sources", []),
                    "used_predict": result.get("used_predict", False)
                }

                # Show details
                with st.expander("Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Intent", metadata["intent"])
                    with col2:
                        st.metric("Chunks Retrieved", metadata["num_chunks"])
                    with col3:
                        st.metric("Response Time", f"{metadata['latency_ms']/1000:.1f}s")

                    if metadata["sources"]:
                        st.markdown("### Sources")
                        display_sources(metadata["sources"])

                # Store assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": metadata
                })

            except Exception as e:
                err = f"Error: {str(e)}"
                st.error(err)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err
                })

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("""
        This assistant helps you search and query your Gmail inbox using RAG.
    """)

    with st.expander("Features"):
        st.markdown("""
        - Natural language queries  
        - Event/calendar detection  
        - Deadline tracking  
        - Smart intent routing  
        - Hybrid search (BM25 + Dense)  
        - Query reranking  
        """)

    st.divider()
    st.header("Session Stats")
    st.metric("Messages", len(st.session_state.messages))
    st.caption(f"Session: {st.session_state.session_id[:8]}...")

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()

    with st.expander("System Info"):
        try:
            rag_temp = RAGQuery()
            st.markdown(f"**Vector DB:** {rag_temp.vector_db.count()} chunks")
            if rag_temp.predict:
                st.markdown(f"**Events DB:** {rag_temp.predict.count()} events")
            st.markdown(f"**Router:** {'Yes' if rag_temp.router else 'No'}")
            st.markdown(f"**Memory:** {'Yes' if rag_temp.memory else 'No'}")
            st.markdown(f"**Hybrid Search:** {'Yes' if rag_temp.enable_hybrid_retrieval else 'No'}")
            st.markdown(f"**Reranking:** {'Yes' if rag_temp.enable_reranking else 'No'}")
        except Exception as e:
            st.error(f"Could not load: {e}")
