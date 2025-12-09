"""
Streamlit UI for Gmail Assistant RAG System
--------------------------------------------
Streamlined UI: User configures settings, pipeline runs automatically.
"""

import streamlit as st
import sys
import os
import yaml
from datetime import datetime, date

# Add parent directory to path for imports
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


# Page configuration
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
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-ready { background-color: #28a745; }
    .status-processing { background-color: #ffc107; }
    .status-error { background-color: #dc3545; }
    .stButton>button[kind="primary"] {
        background-color: #1f77b6;
        border-color: #1f77b6;
        color: white;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #1a6ba5;
        border-color: #1a6ba5;
    }
    /* Purple button for Refresh - using key selector */
    div[data-testid="stButton"]:has(button[key="refresh_emails"]) button {
        background-color: #9b59b6 !important;
        color: white !important;
        border-color: #9b59b6 !important;
    }
    div[data-testid="stButton"]:has(button[key="refresh_emails"]) button:hover {
        background-color: #8e44ad !important;
        border-color: #8e44ad !important;
    }
    /* Green button for Connect Gmail - using key selector */
    div[data-testid="stButton"]:has(button[key="main_connect_gmail"]) button {
        background-color: #28a745 !important;
        color: white !important;
        border-color: #28a745 !important;
        font-size: 1.1rem !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: bold !important;
    }
    div[data-testid="stButton"]:has(button[key="main_connect_gmail"]) button:hover {
        background-color: #218838 !important;
        border-color: #218838 !important;
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
if "last_config_hash" not in st.session_state:
    st.session_state.last_config_hash = None
if "initial_config_hash" not in st.session_state:
    st.session_state.initial_config_hash = None
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None


def load_config():
    """Load configuration file."""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Failed to load config: {str(e)}")
        return None


def save_config(config):
    """Save configuration to file."""
    try:
        with open("config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        st.error(f"Failed to save config: {str(e)}")
        return False


def get_config_hash(config):
    """Get a hash of config for change detection."""
    gmail_cfg = config.get("gmail", {})
    return hash((
        tuple(gmail_cfg.get("senders", [])),
        gmail_cfg.get("start_date"),
        gmail_cfg.get("end_date"),
        gmail_cfg.get("label")
    ))


def check_gmail_connection(config):
    """Check if Gmail is connected."""
    try:
        token_path = config["creds"]["gmail_token"]
        client_path = config["creds"]["gmail_client"]
        
        if not os.path.exists(client_path):
            return False, "Gmail credentials file not found"
        
        if not os.path.exists(token_path):
            return False, "Not authenticated"
        
        creds = Credentials.from_authorized_user_file(token_path, ['https://www.googleapis.com/auth/gmail.readonly'])
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    with open(token_path, 'w') as token:
                        token.write(creds.to_json())
                    return True, "Connected (refreshed)"
                except:
                    return False, "Token expired, re-authentication needed"
            else:
                return False, "Not authenticated"
        
        return True, "Connected"
    except Exception as e:
        return False, f"Error: {str(e)}"


def connect_gmail(config):
    """Connect to Gmail (OAuth flow)."""
    try:
        client_path = config["creds"]["gmail_client"]
        token_path = config["creds"]["gmail_token"]
        
        if not os.path.exists(client_path):
            st.error("Gmail credentials file not found. Please add your OAuth client credentials to creds/gmail_creds.json")
            return False
        
        creds = None
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, ['https://www.googleapis.com/auth/gmail.readonly'])
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    client_path, ['https://www.googleapis.com/auth/gmail.readonly'])
                creds = flow.run_local_server(port=0, open_browser=True)
            
            os.makedirs(os.path.dirname(token_path), exist_ok=True)
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        
        # Test connection
        service = build('gmail', 'v1', credentials=creds)
        profile = service.users().getProfile(userId='me').execute()
        st.session_state.gmail_connected = True
        return True
    except Exception as e:
        st.error(f"Failed to connect to Gmail: {str(e)}")
        return False


def run_full_pipeline(config, status_container=None, progress_bar=None):
    """Run the complete email processing pipeline.
    
    Databases are created automatically when modules are initialized:
    - emails.db: Created when DBHelper is first used
    - events.db: Created when Predict module is initialized (in chunker)
    - memory.db: Created when Memory module is initialized (in RAGQuery)
    - Vector DB: Created when EmailVectorDB is initialized
    """
    try:
        if not st.session_state.gmail_connected:
            if status_container:
                status_container.warning("⚠️ Gmail not connected. Skipping ingestion.")
            return False
        
        # Step 1: Gmail Ingestion (creates emails.db via DBHelper)
        if status_container:
            status_container.text("Step 1/5: Fetching emails from Gmail...")
        if progress_bar:
            progress_bar.progress(0.2)  # 20%
        ingestor = GmailIngestor(config_path="config.yaml")
        ingestor.run()
        
        # Step 2: Docling Processing
        if status_container:
            status_container.text("Step 2/5: Processing emails with Docling...")
        if progress_bar:
            progress_bar.progress(0.4)  # 40%
        processor = DoclingProcessor(config_path="config.yaml")
        processor.process_all_emails()
        
        # Step 3: Chunking (creates events.db via Predict module if enabled)
        if status_container:
            status_container.text("Step 3/5: Chunking emails...")
        if progress_bar:
            progress_bar.progress(0.6)  # 60%
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
            progress_bar.progress(0.8)  # 80%
        from src.embedder import EmailEmbedder
        embedder = EmailEmbedder(config_path="config.yaml", db_helper=db)
        embedded_chunks = embedder.embed_all_messages()
        
        # Step 5: Indexing (creates vector DB)
        if status_container:
            status_container.text("Step 5/5: Indexing in vector database...")
        if progress_bar:
            progress_bar.progress(0.95)  # 95%
        vector_db = EmailVectorDB(config_path="config.yaml")
        if embedded_chunks:
            vector_db.add_chunks(embedded_chunks)
        
        db.close()
        
        if progress_bar:
            progress_bar.progress(1.0)  # 100%
        
        if status_container:
            status_container.success("✅ Pipeline completed successfully!")
        
        # No need to clear RAG system cache - we create it fresh for each query
        
        return True
    except Exception as e:
        if status_container:
            status_container.error(f"❌ Pipeline error: {str(e)}")
        import traceback
        st.error(f"Pipeline error: {str(e)}")
        st.code(traceback.format_exc())
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


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<div class="main-header">📧 Gmail Assistant</div>', unsafe_allow_html=True)
    
    # Load config
    config = load_config()
    if config is None:
        st.error("Failed to load configuration. Please check config.yaml")
        st.stop()
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Gmail Connection Status (just status, button is in main area)
        st.subheader("🔐 Gmail Connection")
        is_connected, status_msg = check_gmail_connection(config)
        
        if is_connected:
            st.success(f"✅ {status_msg}")
            st.session_state.gmail_connected = True
        else:
            st.warning(f"⚠️ {status_msg}")
            st.session_state.gmail_connected = False
        
        st.divider()
        
        # Email Settings
        st.subheader("📧 Email Settings")
        
        # Senders
        st.write("**Senders**")
        senders = config.get("gmail", {}).get("senders", [])
        new_senders = st.text_area(
            "Email addresses (one per line)",
            value="\n".join(senders),
            help="Enter one email address per line. Use 'all' to fetch from all senders.",
            height=100
        )
        config["gmail"]["senders"] = [s.strip() for s in new_senders.split("\n") if s.strip()]
        
        # Date filters
        st.write("**Date Range**")
        col1, col2 = st.columns(2)
        with col1:
            start_date_str = config["gmail"].get("start_date", "")
            start_date_val = None
            if start_date_str:
                try:
                    start_date_val = datetime.strptime(start_date_str, "%Y-%m-%d").date()
                except:
                    pass
            start_date = st.date_input(
                "Start Date",
                value=start_date_val,
                help="Leave empty to ignore"
            )
        with col2:
            end_date_str = config["gmail"].get("end_date", "")
            end_date_val = None
            if end_date_str:
                try:
                    end_date_val = datetime.strptime(end_date_str, "%Y-%m-%d").date()
                except:
                    pass
            end_date = st.date_input(
                "End Date",
                value=end_date_val,
                help="Leave empty to ignore"
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
        st.write("**Gmail Label**")
        label = st.text_input(
            "Label",
            value=config["gmail"].get("label", "INBOX"),
            help="Gmail label to target: INBOX, UNREAD, STARRED, etc."
        )
        config["gmail"]["label"] = label
            
        # Initialize config hash tracking
        current_config_hash = get_config_hash(config)
        if st.session_state.initial_config_hash is None:
            st.session_state.initial_config_hash = current_config_hash
            st.session_state.last_config_hash = current_config_hash
        
        # Check if config has changed from initial state
        config_changed = current_config_hash != st.session_state.initial_config_hash
        
        if config_changed:
            st.info("💡 Configuration changed. Click 'Save & Process' to update.")
        
        if st.button("💾 Save & Process", use_container_width=True, type="primary"):
            if save_config(config):
                st.session_state.last_config_hash = get_config_hash(config)
                st.session_state.initial_config_hash = get_config_hash(config)  # Reset initial hash
                st.session_state.pipeline_running = True
                st.info("Starting pipeline...")
                st.rerun()
        
        # Manual refresh button with purple styling
        if st.button("🔄 Refresh Emails", use_container_width=True, help="Fetch new emails and reprocess", key="refresh_emails"):
            if not st.session_state.gmail_connected:
                st.warning("Please connect to Gmail first")
            else:
                st.session_state.pipeline_running = True
                st.info("Starting refresh...")
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
    
    # Main content area
    # Prominent Gmail Connection Button
    is_connected, status_msg = check_gmail_connection(config)
    
    if not is_connected:
        st.session_state.gmail_connected = False
        st.markdown("### 🔐 Connect to Gmail")
        st.markdown("Connect your Gmail account to get started. This is the first step to begin processing your emails.")
        
        # Large green connect button - using HTML for custom styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <style>
            .connect-gmail-btn {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 0.75rem 2rem;
                font-size: 1.2rem;
                font-weight: bold;
                border-radius: 0.5rem;
                cursor: pointer;
                width: 100%;
                text-align: center;
            }
            .connect-gmail-btn:hover {
                background-color: #218838;
            }
            </style>
            """, unsafe_allow_html=True)
            
            if st.button("🔗 Connect to Gmail", use_container_width=True, key="main_connect_gmail", type="primary"):
                with st.spinner("Connecting to Gmail..."):
                    if connect_gmail(config):
                        st.success("✅ Connected successfully!")
                        st.session_state.gmail_connected = True
                        # Trigger pipeline after connection
                        st.session_state.pipeline_running = True
                        st.rerun()
        
        st.divider()
    else:
        st.session_state.gmail_connected = True
    
    # Check if pipeline needs to run
    if st.session_state.pipeline_running:
        st.session_state.pipeline_running = False
        st.info("🔄 Processing emails... This may take a few minutes.")
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        success = run_full_pipeline(config, status_placeholder, progress_bar)
        
        if success:
            st.success("✅ Pipeline completed! You can now ask questions.")
        else:
            st.error("❌ Pipeline failed. Please check the error messages above.")
        
        st.rerun()
    
    # Query Interface
    st.header("Ask Questions About Your Emails")
    
    # Check if system is ready
    stats = get_system_stats(config)
    if not stats or stats["vector_count"] == 0:
        st.warning("⚠️ No emails indexed yet. Please configure settings and click 'Save & Process' to start.")
        st.info("💡 The system will automatically fetch emails, process them, and make them searchable.")
    else:
        # Note: We don't cache RAG system because it contains database connections
        # that can't be reused across Streamlit reruns (different threads)
        
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
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your emails..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching emails and generating answer..."):
                    rag_system = None
                    try:
                        # Create RAG system fresh for each query to avoid thread issues
                        # SQLite connections can't be reused across Streamlit reruns (different threads)
                        rag_system = RAGQuery(config_path="config.yaml")
                        result = rag_system.query(prompt, top_k=top_k)
                        
                        # Display answer
                        answer = result.get("answer", "No answer generated.")
                        st.markdown(answer)
                        
                        # Prepare metadata
                        metadata = {
                            "num_sources": len(result.get("sources", [])),
                            "num_chunks": result.get("num_chunks_retrieved", 0)
                        }
                        
                        if "intent" in result:
                            metadata["intent"] = result["intent"]
                            st.caption(f"Intent: **{result['intent']}** | Sources: {len(result.get('sources', []))}")
                        
                        # Display sources
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
                        
                        # Add to chat history
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
                        # Always close connections to avoid thread issues
                        if rag_system is not None:
                            try:
                                rag_system.close()
                            except:
                                pass
        
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
                    if st.button(example, key=f"example_{i}", use_container_width=True):
                        st.session_state.messages.append({"role": "user", "content": example})
                        st.rerun()


if __name__ == "__main__":
    main()
