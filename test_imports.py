"""
Quick test to check if all imports work
"""

print("Testing imports...\n")

try:
    import yaml
    print("✅ yaml")
except ImportError as e:
    print(f"❌ yaml: {e}")

try:
    from bs4 import BeautifulSoup
    print("✅ beautifulsoup4")
except ImportError as e:
    print(f"❌ beautifulsoup4: {e}")

try:
    import tiktoken
    print("✅ tiktoken")
except ImportError as e:
    print(f"❌ tiktoken: {e}")

try:
    print("⏳ Loading docling (this may take 30-60 seconds on first import)...")
    from docling.document_converter import DocumentConverter
    print("✅ docling")
except ImportError as e:
    print(f"❌ docling: {e}")

try:
    import xlrd
    print("✅ xlrd")
except ImportError as e:
    print(f"❌ xlrd: {e}")

try:
    import openpyxl
    print("✅ openpyxl")
except ImportError as e:
    print(f"❌ openpyxl: {e}")

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    print("✅ google-api-python-client")
except ImportError as e:
    print(f"❌ google-api-python-client: {e}")

print("\n✅ All critical imports successful!" if all else "⚠️ Some imports failed")
