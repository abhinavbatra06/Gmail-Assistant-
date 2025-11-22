import os, json, base64, email, yaml
from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from datetime import datetime
from src.storage_manager import StorageManager
from src.db_helper import DBHelper


class GmailIngestor:
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

    def __init__(self, config_path="config.yaml"):
        # 1. Load config + setup storage + DB
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.paths = StorageManager(config_path)
        self.db = DBHelper(self.paths.db_path)
        self.gmail_cfg = self.cfg["gmail"]
        self.creds_cfg = self.cfg["creds"]
        self.service = self._get_gmail_service()

    # ------------------------------------------------------------------
    def _get_gmail_service(self):
        creds = None
        token_path = self.creds_cfg["gmail_token"]
        client_path = self.creds_cfg["gmail_client"]

        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, self.SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(client_path, self.SCOPES)
                creds = flow.run_local_server(port=0)
            os.makedirs(os.path.dirname(token_path), exist_ok=True)
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        return build('gmail', 'v1', credentials=creds)

    # ------------------------------------------------------------------
    def _decode(self, data_b64):
        return base64.urlsafe_b64decode(data_b64).decode('utf-8', errors='ignore')

    def _extract_bodies(self, payload):
        text, html = "", ""
        mime = payload.get("mimeType", "")
        body = payload.get("body", {})
        data = body.get("data")

        if data:
            decoded = self._decode(data)
            if mime == "text/plain": text = decoded
            elif mime == "text/html": html = decoded

        for part in payload.get("parts", []) or []:
            t, h = self._extract_bodies(part)
            text += f"\n{t}"
            html += f"\n{h}"
        return text.strip(), html.strip()

    def _save_attachment(self, msg_id, part):
        att_id = part['body']['attachmentId']
        att = self.service.users().messages().attachments().get(
            userId='me', messageId=msg_id, id=att_id).execute()
        data = base64.urlsafe_b64decode(att['data'])
        fname = part.get('filename', f"{att_id}.bin")
        path = self.paths.path_for_attachment(msg_id, fname)
        with open(path, "wb") as f:
            f.write(data)
        return fname, path

    # ------------------------------------------------------------------
    def _build_query(self, sender):
        q_parts = []
        if sender.lower() != "all":
            q_parts.append(f"from:{sender}")
        if self.gmail_cfg.get("start_date"):
            q_parts.append(f"after:{self.gmail_cfg['start_date'].replace('-', '/')}")
        if self.gmail_cfg.get("end_date"):
            q_parts.append(f"before:{self.gmail_cfg['end_date'].replace('-', '/')}")
        if self.gmail_cfg.get("label"):
            q_parts.append(f"label:{self.gmail_cfg['label']}")
        return " ".join(q_parts)

    # ------------------------------------------------------------------
    def run(self):
        senders = self.gmail_cfg.get("senders", ["all"])
        target_total = self.gmail_cfg.get("target_total", 50)
        per_sender_cap = self.gmail_cfg.get("per_sender_cap", 10)
        include_attachments = self.gmail_cfg.get("include_attachments", True)

        total_saved = 0
        for sender in senders:
            if total_saved >= target_total: break

            query = self._build_query(sender)
            remaining = target_total - total_saved
            max_fetch = min(per_sender_cap, remaining)

            res = self.service.users().messages().list(
                userId='me', q=query, maxResults=max_fetch).execute()

            for msg_meta in res.get('messages', []):
                msg_id = msg_meta['id']
                if self.db.email_exists(msg_id):
                    continue

                # get full message
                msg = self.service.users().messages().get(userId='me', id=msg_id, format='full').execute()
                payload = msg.get('payload', {})
                headers = payload.get('headers', [])
                subject = sender_line = date = ""

                for h in headers:
                    n = h.get('name', '')
                    if n == 'Subject': subject = h.get('value', '')
                    elif n == 'From': sender_line = h.get('value', '')
                    elif n == 'Date': date = h.get('value', '')

                # bodies
                text_body, html_body = self._extract_bodies(payload)

                # save raw EML
                raw_msg = self.service.users().messages().get(userId='me', id=msg_id, format='raw').execute()
                eml_data = base64.urlsafe_b64decode(raw_msg['raw'].encode('UTF-8'))
                eml_path = self.paths.path_for_email(msg_id)
                with open(eml_path, "wb") as f:
                    f.write(eml_data)

                # attachments
                attachments = []
                if include_attachments:
                    for p in payload.get("parts", []) or []:
                        if p.get("filename") and p["body"].get("attachmentId"):
                            fname, fpath = self._save_attachment(msg_id, p)
                            attachments.append({"filename": fname, "path": fpath})

                # metadata json
                meta = {
                    "id": msg_id,
                    "from": sender_line,
                    "subject": subject,
                    "date": date,
                    "eml_path": eml_path,
                    "attachments": attachments,
                    "body_text": text_body,
                    "body_html": html_body
                }
                meta_path = self.paths.path_for_metadata(msg_id)
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                # DB insert
                self.db.insert_email(msg_id, sender_line, subject, date, eml_path, meta_path)
                total_saved += 1

                print(f"✅ Saved email {msg_id} | {subject[:60]}")

        print(f"\n🎯 Done. Total emails saved: {total_saved}")
        self.db.close()

if __name__ == "__main__":
    print("Starting GmailIngestor...")  # simple debug
    ingestor = GmailIngestor(config_path="config.yaml")
    ingestor.run()
