"""
Gmail Assistant - Streamlit Chat UI
Simple chat interface for querying emails
"""
import streamlit as st
import time
from src.rag_query import RAGQuery

# Page config
st.set_page_config(
    page_title="Gmail Assistant",
    page_icon="",
    layout="wide"
)

# Title
st.title("Gmail Assistant")
st.caption("Ask questions about your emails")

# Note: RAGQuery creates fresh DB connections internally to avoid threading issues

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show metadata for assistant messages
        if message["role"] == "assistant" and "metadata" in message:
            with st.expander("Details"):
                metadata = message["metadata"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Intent", metadata.get("intent", "N/A"))
                with col2:
                    st.metric("Chunks", metadata.get("num_chunks", 0))
                with col3:
                    st.metric("Latency", f"{metadata.get('latency_ms', 0)/1000:.1f}s")
                
                # Show sources
                if "sources" in metadata and metadata["sources"]:
                    st.write("**Sources:**")
                    for source in metadata["sources"][:3]:
                        st.code(source, language=None)

# Chat input
if prompt := st.chat_input("Ask about your emails..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching emails..."):
            start_time = time.time()
            
            try:
                # Create fresh RAG instance for this query to avoid SQLite threading issues
                rag = RAGQuery()
                result = rag.query(prompt)
                
                # Extract response
                answer = result.get("answer", "Sorry, I couldn't find an answer.")
                
                # Display answer
                st.markdown(answer)
                
                # Prepare metadata
                metadata = {
                    "intent": result.get("intent", "N/A"),
                    "num_chunks": result.get("num_chunks_retrieved", 0),
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "sources": result.get("sources", [])
                }
                
                # Show details in expander
                with st.expander("Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Intent", metadata["intent"])
                    with col2:
                        st.metric("Chunks", metadata["num_chunks"])
                    with col3:
                        st.metric("Latency", f"{metadata['latency_ms']/1000:.1f}s")
                    
                    # Show sources
                    if metadata["sources"]:
                        st.write("**Sources:**")
                        for source in metadata["sources"][:3]:
                            st.code(source, language=None)
                
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": metadata
                })
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("This assistant helps you search and query your Gmail inbox using RAG.")
    
    st.divider()
    
    st.header("Stats")
    st.metric("Messages in chat", len(st.session_state.messages))
    
    st.divider()
    
    # Clear chat button
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
