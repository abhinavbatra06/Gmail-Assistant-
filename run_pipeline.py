import sys, os
sys.path.append(os.path.dirname(__file__))

from src.gmail_ingest import GmailIngestor

def main():
    print("🚀 Starting Gmail ingestion pipeline...")
    ingestor = GmailIngestor("config.yaml")
    ingestor.run()
    print("\n✅ Gmail ingestion completed successfully.")

if __name__ == "__main__":
    main()
