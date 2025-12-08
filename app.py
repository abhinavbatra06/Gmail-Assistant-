"""
Gmail Assistant - Streamlit UI
Clean, streamlined interface for email RAG system
"""

import streamlit as st
import sys
import os
import yaml
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.gmail_ingest import GmailIngestor
from src.docling_processor import DoclingProcessor
from src.db_helper import DBHelper
from src.rag_query import RAGQuery
from src.vector_db import EmailVectorDB
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Page config
st.set_page_config(
    page_title="Gmail Assistant",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .connect-button {
        background-color: #28a745 !important;
        color: white !important;
        border-color: #28a745 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "gmail_connected" not in st.session_state:
    st.session_state.gmail_connected = False
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "query_processing" not in st.session_state:
    st.session_state.query_processing = False
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


def load_config():
    """Load configuration from YAML file."""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Failed to load config: {str(e)}")
        return None


def save_config(config):
    """Save configuration to YAML file."""
    try:
        with open("config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        st.error(f"Failed to save config: {str(e)}")
        return False


def check_gmail_connection(config):
    """Check if Gmail is connected."""
    token_path = config.get("creds", {}).get("gmail_token", "creds/token.json")
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(
                token_path, 
                ['https://www.googleapis.com/auth/gmail.readonly']
            )
            if creds and creds.valid:
                return True, "Connected"
            elif creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
                return True, "Connected"
        except:
            pass
    return False, "Not connected"


def connect_gmail(config):
    """Connect to Gmail via OAuth."""
    try:
        token_path = config.get("creds", {}).get("gmail_token", "creds/token.json")
        client_path = config.get("creds", {}).get("gmail_client", "creds/gmail_creds.json")
        
        creds = None
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(
                token_path, 
                ['https://www.googleapis.com/auth/gmail.readonly']
            )
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    client_path, 
                    ['https://www.googleapis.com/auth/gmail.readonly']
                )
                creds = flow.run_local_server(port=0, open_browser=True)
            
            os.makedirs(os.path.dirname(token_path), exist_ok=True)
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        
        # Test connection
        service = build('gmail', 'v1', credentials=creds)
        service.users().getProfile(userId='me').execute()
        return True
    except Exception as e:
        st.error(f"Failed to connect: {str(e)}")
        return False


def get_system_stats(config):
    """Get system statistics."""
    try:
        db = DBHelper(config["paths"]["db_path"])
        stats = db.get_chunking_stats()
        embedding_stats = db.get_embedding_stats()
        db.close()
        
        vector_db = EmailVectorDB(config_path="config.yaml")
        vector_count = vector_db.count()
        
        return {
            "total_emails": stats.get("total_emails", 0),
            "chunked_emails": stats.get("chunked_emails", 0),
            "total_chunks": stats.get("total_chunks", 0),
            "embedded_emails": embedding_stats.get("embedded_emails", 0),
            "total_embeddings": embedding_stats.get("total_embeddings", 0),
            "vector_count": vector_count
        }
    except Exception as e:
        return None


def run_pipeline(config, status_container=None, progress_bar=None):
    """Run the full email processing pipeline."""
    try:
        if not st.session_state.gmail_connected:
            if status_container:
                status_container.error("❌ Gmail not connected")
            return False
        
        # Step 1: Gmail Ingestion
        if status_container:
            status_container.text("Step 1/5: Fetching emails from Gmail...")
        if progress_bar:
            progress_bar.progress(0.2)
        ingestor = GmailIngestor(config_path="config.yaml")
        ingestor.run()
        
        # Step 2: Docling Processing
        if status_container:
            status_container.text("Step 2/5: Processing emails with Docling...")
        if progress_bar:
            progress_bar.progress(0.4)
        processor = DoclingProcessor(config_path="config.yaml")
        processor.process_all_emails()
        
        # Step 3: Chunking
        if status_container:
            status_container.text("Step 3/5: Chunking emails...")
        if progress_bar:
            progress_bar.progress(0.6)
        from src.chunker import EmailChunker
        db = DBHelper(config["paths"]["db_path"])
        chunking_config = config.get("chunking", {})
        chunk_size = chunking_config.get("chunk_size", 600)
        overlap = chunking_config.get("chunk_overlap", 100)
        rag_config = config.get("rag", {})
        predict_db_path = rag_config.get("predict_db_path", "db/events.db") if rag_config.get("enable_predict", True) else None
        
        chunker = EmailChunker(
            db_helper=db,
            chunk_size=chunk_size,
            overlap=overlap,
            predict_db_path=predict_db_path
        )
        chunker.chunk_all_emails()
        
        # Step 4: Embedding
        if status_container:
            status_container.text("Step 4/5: Generating embeddings...")
        if progress_bar:
            progress_bar.progress(0.8)
        
        embedded_chunks = []
        try:
            from src.embedder import EmailEmbedder
            embedder = EmailEmbedder(config_path="config.yaml", db_helper=db)
            embedded_chunks = embedder.embed_all_messages()
            if status_container:
                if embedded_chunks:
                    status_container.text(f"Step 4/5: Generated {len(embedded_chunks)} embeddings")
                else:
                    status_container.warning("⚠️ No new embeddings generated")
        except Exception as e:
            if status_container:
                status_container.error(f"❌ Embedding failed: {str(e)}")
            st.error(f"Embedding error: {str(e)}")
            raise
        
        # Step 5: Indexing
        if status_container:
            status_container.text("Step 5/5: Indexing in vector database...")
        if progress_bar:
            progress_bar.progress(0.95)
        
        try:
            vector_db = EmailVectorDB(config_path="config.yaml")
            initial_count = vector_db.count()
            
            # Get embedding stats
            embedding_stats = db.get_embedding_stats()
            total_embeddings = embedding_stats.get("total_embeddings", 0)
            vector_count = vector_db.count()
            needs_indexing = vector_count < total_embeddings
            
            # Step 5a: Add new chunks if we have them
            if embedded_chunks:
                if status_container:
                    status_container.text(f"Step 5/5: Adding {len(embedded_chunks)} new chunks to vector DB...")
                
                # Validate chunk format
                sample = embedded_chunks[0]
                required_keys = ["chunk_id", "embedding", "text", "metadata"]
                missing = [k for k in required_keys if k not in sample]
                if missing:
                    error_msg = f"Invalid chunk format. Missing keys: {missing}"
                    if status_container:
                        status_container.error(f"❌ {error_msg}")
                    st.error(error_msg)
                    st.json(sample)
                    raise ValueError(error_msg)
                
                # Add chunks
                vector_db.add_chunks(embedded_chunks, skip_existing=True)
                
                # Refresh stats after adding
                embedding_stats = db.get_embedding_stats()
                vector_count = vector_db.count()
                needs_indexing = vector_count < embedding_stats.get("total_embeddings", 0)
            
            # Step 5b: If embeddings exist but aren't indexed, re-embed and index
            if needs_indexing:
                missing = embedding_stats.get("total_embeddings", 0) - vector_count
                if status_container:
                    status_container.warning(f"⚠️ {missing} embeddings not indexed. Re-embedding to index...")
                
                # Clear embedding status (like terminal script)
                import sqlite3
                conn = sqlite3.connect(config["paths"]["db_path"])
                cur = conn.cursor()
                cur.execute("DELETE FROM embedding_status")
                conn.commit()
                conn.close()
                
                # Re-embed all messages
                from src.embedder import EmailEmbedder
                embedder = EmailEmbedder(config_path="config.yaml", db_helper=db)
                all_chunks = embedder.embed_all_messages()
                
                if all_chunks:
                    if status_container:
                        status_container.text(f"Step 5/5: Indexing {len(all_chunks)} chunks...")
                    vector_db.add_chunks(all_chunks, skip_existing=True)
                    final_count = vector_db.count()
                    if status_container:
                        status_container.text(f"Step 5/5: Indexed {len(all_chunks)} chunks (total: {final_count})")
                else:
                    if status_container:
                        status_container.warning("⚠️ No chunks generated for indexing")
            else:
                # Everything is up to date
                final_count = vector_db.count()
                if status_container:
                    if final_count > 0:
                        status_container.text(f"Step 5/5: Vector DB up to date ({final_count} chunks)")
                    else:
                        status_container.warning("⚠️ No chunks in vector DB")
                        
        except Exception as e:
            if status_container:
                status_container.error(f"❌ Indexing failed: {str(e)}")
            st.error(f"Indexing error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            raise
        
        db.close()
        
        if progress_bar:
            progress_bar.progress(1.0)
        
        # Verify final state
        final_count = vector_db.count()
        if status_container:
            if final_count > 0:
                status_container.success(f"✅ Pipeline completed! {final_count} chunks indexed.")
            else:
                status_container.error("❌ Pipeline completed but no chunks in vector DB")
        
        return final_count > 0
        
    except Exception as e:
        if status_container:
            status_container.error(f"❌ Pipeline error: {str(e)}")
        st.error(f"Pipeline error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False


def render_email_settings(config, container):
    """Render email settings form."""
    # Senders
    container.write("**Senders**")
    senders = config.get("gmail", {}).get("senders", [])
    new_senders = container.text_area(
        "Email addresses (one per line)",
        value="\n".join(senders),
        help="Enter one email address per line. Use 'all' to fetch from all senders.",
        height=100,
        key="senders_input"
    )
    config["gmail"]["senders"] = [s.strip() for s in new_senders.split("\n") if s.strip()]
    
    # Date Range
    container.write("**Date Range**")
    col1, col2 = container.columns(2)
    with col1:
        start_date_str = config["gmail"].get("start_date", "")
        start_date_val = None
        if start_date_str:
            try:
                start_date_val = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            except:
                pass
        start_date = container.date_input(
            "Start Date",
            value=start_date_val,
            help="Leave empty to ignore",
            key="start_date_input"
        )
    with col2:
        end_date_str = config["gmail"].get("end_date", "")
        end_date_val = None
        if end_date_str:
            try:
                end_date_val = datetime.strptime(end_date_str, "%Y-%m-%d").date()
            except:
                pass
        end_date = container.date_input(
            "End Date",
            value=end_date_val,
            help="Leave empty to ignore",
            key="end_date_input"
        )
    
    if start_date:
        config["gmail"]["start_date"] = start_date.strftime("%Y-%m-%d")
    else:
        config["gmail"]["start_date"] = None
    
    if end_date:
        config["gmail"]["end_date"] = end_date.strftime("%Y-%m-%d")
    else:
        config["gmail"]["end_date"] = None
    
    # Gmail Label
    container.write("**Gmail Label**")
    label = container.text_input(
        "Label",
        value=config["gmail"].get("label", "INBOX"),
        help="Gmail label to target: INBOX, UNREAD, STARRED, etc.",
        key="label_input"
    )
    config["gmail"]["label"] = label
    
    return config


def process_query(prompt, top_k, show_sources):
    """Process a query and return response."""
    st.session_state.query_processing = True
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching emails and generating answer..."):
            rag_system = None
            try:
                rag_system = RAGQuery(config_path="config.yaml")
                result = rag_system.query(prompt, top_k=top_k)
                
                answer = result.get("answer", "No answer generated.")
                st.markdown(answer)
                
                metadata = {
                    "num_sources": len(result.get("sources", [])),
                    "num_chunks": result.get("num_chunks_retrieved", 0)
                }
                
                if "intent" in result:
                    metadata["intent"] = result["intent"]
                    st.caption(f"Intent: **{result['intent']}** | Sources: {len(result.get('sources', []))}")
                
                sources = result.get("sources", [])
                if sources and show_sources:
                    st.divider()
                    st.subheader(f"📚 Sources ({len(sources)})")
                    for i, source in enumerate(sources, 1):
                        with st.expander(f"📧 {source.get('subject', 'No subject')}", expanded=False):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**From:** {source.get('from', 'Unknown')}")
                                st.write(f"**Date:** {source.get('date', 'Unknown')}")
                            with col2:
                                st.write(f"**Type:** {source.get('source_type', 'unknown')}")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": metadata,
                    "sources": sources
                })
                
            except Exception as e:
                error_msg = f"❌ Error processing query: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
            finally:
                if rag_system is not None:
                    try:
                        rag_system.close()
                    except:
                        pass
                st.session_state.query_processing = False


def main():
    """Main application."""
    st.markdown('<div class="main-header">📧 Gmail Assistant</div>', unsafe_allow_html=True)
    
    # Load config
    config = load_config()
    if config is None:
        st.stop()
    
    # Check Gmail connection
    is_connected, status_msg = check_gmail_connection(config)
    st.session_state.gmail_connected = is_connected
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("🔐 Gmail Connection")
        if is_connected:
            st.success(f"✅ {status_msg}")
        else:
            st.warning(f"⚠️ {status_msg}")
        
        st.divider()
        
        # Show email settings in sidebar if connected
        if is_connected:
            st.subheader("📧 Email Settings")
            config = render_email_settings(config, st)
            
            if st.button("💾 Save & Process", use_container_width=True, type="primary", key="save_sidebar"):
                if save_config(config):
                    st.session_state.pipeline_running = True
                    st.success("Configuration saved!")
                    st.rerun()
            
            if st.button("🔄 Refresh Emails", use_container_width=True, key="refresh_sidebar"):
                st.session_state.pipeline_running = True
                st.rerun()
        
        st.divider()
        
        # System Status
        st.subheader("📊 System Status")
        stats = get_system_stats(config)
        if stats:
            st.metric("Total Emails", stats["total_emails"])
            st.metric("Indexed", stats["vector_count"])
            if stats["vector_count"] > 0:
                st.success("✅ Ready for queries")
            else:
                st.warning("⚠️ No emails indexed yet")
        else:
            st.info("No data yet")
    
    # Main content
    if not is_connected:
        # Setup interface - show settings prominently
        st.markdown("### 📧 Configure Your Email Settings")
        st.markdown("Set up your email preferences before connecting to Gmail.")
        
        with st.container():
            st.markdown("---")
            config = render_email_settings(config, st)
            st.markdown("---")
        
        st.markdown("### 🔐 Connect to Gmail")
        st.markdown("Once you've configured your settings, connect your Gmail account to start processing emails.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔗 Connect to Gmail", use_container_width=True, key="connect_main", type="primary"):
                # Save config first
                save_config(config)
                with st.spinner("Connecting to Gmail..."):
                    if connect_gmail(config):
                        st.success("✅ Connected successfully!")
                        st.session_state.gmail_connected = True
                        st.session_state.pipeline_running = True
                        st.rerun()
        
        st.divider()
    else:
        # Pipeline execution
        if st.session_state.pipeline_running:
            st.session_state.pipeline_running = False
            st.info("🔄 Processing emails... This may take a few minutes.")
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            success = run_pipeline(config, status_placeholder, progress_bar)
            
            if success:
                st.success("✅ Pipeline completed! You can now ask questions.")
            else:
                st.error("❌ Pipeline failed. Please check the error messages above.")
            
            st.rerun()
        
        # Query Interface
        st.header("Ask Questions About Your Emails")
        
        stats = get_system_stats(config)
        if not stats or stats["vector_count"] == 0:
            st.warning("⚠️ No emails indexed yet. Please configure settings and click 'Save & Process' to start.")
            st.info("💡 The system will automatically fetch emails, process them, and make them searchable.")
        else:
            # Query settings
            col1, col2 = st.columns([4, 1])
            with col1:
                top_k = st.slider("Number of sources to retrieve", min_value=1, max_value=20, value=5)
            with col2:
                show_sources = st.checkbox("Show Sources", value=True)
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and "metadata" in message:
                        metadata = message["metadata"]
                        if "intent" in metadata:
                            st.caption(f"Intent: **{metadata['intent']}** | Sources: {metadata.get('num_sources', 0)}")
            
            # Processing status
            if st.session_state.query_processing:
                st.warning("⏳ Processing query... Please wait.")
            
            # Check for pending query from example buttons
            if st.session_state.pending_query and not st.session_state.query_processing:
                prompt = st.session_state.pending_query
                st.session_state.pending_query = None
                process_query(prompt, top_k, show_sources)
                st.rerun()
            
            # Chat input
            if st.session_state.query_processing:
                st.chat_input("⏳ Processing query... Please wait.", disabled=True)
            else:
                if prompt := st.chat_input("Ask a question about your emails..."):
                    process_query(prompt, top_k, show_sources)
            
            # Example queries
            with st.expander("💡 Example Queries", expanded=False):
                examples = [
                    "What events are happening this week?",
                    "Show me emails about assignments",
                    "What are my upcoming deadlines?",
                    "Find emails from NYU",
                    "What meetings do I have?"
                ]
                cols = st.columns(3)
                for i, example in enumerate(examples):
                    with cols[i % 3]:
                        if st.button(
                            example,
                            key=f"example_{i}",
                            use_container_width=True,
                            disabled=st.session_state.query_processing
                        ):
                            if not st.session_state.query_processing:
                                st.session_state.pending_query = example
                                st.rerun()


if __name__ == "__main__":
    main()
