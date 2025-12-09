## Email RAG Assistant

This project enables precise, GenAI-powered answers to natural language queries over your institutional email corpus, using customizable chunking, retrieval, and embedding workflows.

Why this project?
Institutional email is one of the richest sources of unstructured data. With the latest GenAI models, we can extract precise answers for natural language user queries—provided we have high-quality retrieval. This project gives full control over chunking, retrieval, and embeddings to optimize performance for academic email workflows.

Most existing tools are closed, non-customizable systems with limited transparency into retrieval and chunking logic. They don’t allow experimentation with chunking, embeddings, ranking, etc.

## Replication Steps

```bash
# Replication Steps

# 0. Clone the repository
git clone https://github.com/abhinavbatra06/Gmail-Assistant-.git
cd Gmail-Assistant-

# 1. Create virtual environment & activate it
python -m venv .venv
source .venv/bin/activate   # On Windows use: .venv\Scripts\Activate.ps1

# 2. Install requirements
pip install -r requirements.txt

# 3. Ensure .env file exists with your OpenAI API key
# Example: OPENAI_API_KEY=your_key

# 4. Check creds folder for gmail_creds.json
# (Place your Gmail credentials in creds/gmail_creds.json)

# 5. Delete any existing data, db, and __pycache__ folders
rm -rf data/ db/ **/__pycache__
# Optionally delete creds/token.json and reauthenticate when running gmail_ingest.py

# 6. Make sure .gitignore contains .venv/, data/, db/ and other sensitive folders

# 7. Edit config.yaml to set your senders and date range
# (Change the list of senders and date range as needed for your workflow)

# 8. Run the pipeline
python -m src.gmail_ingest
python -m src.docling_processor
python scripts/chunk_all.py   # Or: python -m scripts.chunk_all
python scripts/embed_and_index.py   # Or: python -m scripts.embed_and_index
python -m src.rag_query --query "Your Query"
streamlit run app.py
# After running, open http://localhost:8501 in your browser to use the app.

# 9. Test with queries to check result quality

