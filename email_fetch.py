from __future__ import print_function
import os, json, base64
from bs4 import BeautifulSoup

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # NOTE: put your Google OAuth client JSON as creds.json
            flow = InstalledAppFlow.from_client_secrets_file('creds.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def _decode_part(data_b64):
    if not data_b64: return ""
    return base64.urlsafe_b64decode(data_b64).decode('utf-8', errors='ignore')

def _extract_body(payload):
    """
    Tries plain text first, falls back to HTML→text.
    Handles simple multipart; good enough for a quick proto.
    """
    if not payload: return ""
    mime = payload.get('mimeType', '')
    body = payload.get('body', {})
    data = body.get('data')

    # Direct single-part
    if data and mime == 'text/plain':
        return _decode_part(data)
    if data and mime == 'text/html':
        html = _decode_part(data)
        return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)

    # Multipart (first try text/plain, then text/html)
    parts = payload.get('parts', []) or []
    txt = ""
    for p in parts:
        if p.get('mimeType') == 'text/plain':
            txt = _extract_body(p)
            if txt: return txt
    for p in parts:
        if p.get('mimeType') == 'text/html':
            txt = _extract_body(p)
            if txt: return txt
    return ""

def fetch_from_senders(senders, target_total=50, per_sender_cap=50):
    service = get_gmail_service()
    all_emails = []
    for sender in senders:
        if len(all_emails) >= target_total: break
        remaining = target_total - len(all_emails)
        max_fetch = min(per_sender_cap, remaining)
        q = f'from:{sender}'
        res = service.users().messages().list(userId='me', q=q, maxResults=max_fetch).execute()
        messages = res.get('messages', [])
        print(f"[{sender}] fetched {len(messages)}")

        for m in messages:
            msg = service.users().messages().get(userId='me', id=m['id']).execute()
            payload = msg.get('payload', {})
            headers = payload.get('headers', [])
            subject = sender_line = date = ""
            for h in headers:
                n = h.get('name')
                if n == 'Subject': subject = h.get('value', '')
                elif n == 'From': sender_line = h.get('value', '')
                elif n == 'Date': date = h.get('value', '')
            body = _extract_body(payload).strip()
            all_emails.append({
                "id": m['id'],
                "from": sender_line,
                "subject": subject,
                "date": date,
                "body": body
            })
            if len(all_emails) >= target_total: break
    return all_emails

if __name__ == "__main__":
    # >>> EDIT THIS LIST <<<
    SENDERS = [
        "nyu536@nyu.edu","zl2068@nyu.edu" , "cds-masters@nyu.edu" , 
        "nyuweekly@nyu.edu","entrepreneur@nyu.edu","kangeles@nyu.edu",
        "cdswidseboard@nyu.edu","fj6@nyu.edu","tb116@nyu.edu"        # add more senders if you like
    ]
    emails = fetch_from_senders(SENDERS, target_total=100, per_sender_cap=10)

    with open("emails.json", "w", encoding="utf-8") as f:
        json.dump(emails, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(emails)} emails to emails.json")