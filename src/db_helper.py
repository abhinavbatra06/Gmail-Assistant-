import sqlite3
from datetime import datetime

class DBHelper:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            id TEXT PRIMARY KEY,
            sender TEXT,
            subject TEXT,
            date TEXT,
            eml_path TEXT,
            metadata_path TEXT,
            status TEXT DEFAULT 'downloaded',
            fetched_at TEXT
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS attachments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT,
            filename TEXT,
            path TEXT
        )""")
        self.conn.commit()

    def email_exists(self, msg_id):
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM emails WHERE id=?", (msg_id,))
        return cur.fetchone() is not None

    def insert_email(self, msg_id, sender, subject, date, eml_path, meta_path):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT OR IGNORE INTO emails
            (id, sender, subject, date, eml_path, metadata_path, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (msg_id, sender, subject, date, eml_path, meta_path,
             datetime.now().isoformat()))
        self.conn.commit()

    def update_status(self, msg_id, status):
        cur = self.conn.cursor()
        cur.execute("UPDATE emails SET status=? WHERE id=?", (status, msg_id))
        self.conn.commit()
    
    def insert_attachment(self, message_id, filename, path):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO attachments (message_id, filename, path)
            VALUES (?, ?, ?)""",
            (message_id, filename, path))
        self.conn.commit()
    
    def get_attachments_for_email(self, msg_id):
        cur = self.conn.cursor()
        cur.execute("SELECT filename, path FROM attachments WHERE message_id=?", (msg_id,))
        return cur.fetchall()

    def close(self):
        self.conn.close()
