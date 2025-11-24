"""
One-time migration script to backfill the attachments table from existing metadata JSON files.

This script reads all metadata JSON files and populates the attachments table
for emails that were processed before the attachment tracking was added.

Does not need to be deployed as part of regular application code. Only for one time use to populate the database.
"""

import os
import sys
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db_helper import DBHelper
from src.storage_manager import StorageManager

def migrate_attachments(config_path="config.yaml"):
    """Backfill attachments table from metadata JSON files."""
    
    paths = StorageManager(config_path)
    db = DBHelper(paths.db_path)
    
    metadata_dir = paths.paths["metadata"]
    metadata_files = glob.glob(os.path.join(metadata_dir, "*.json"))
    
    print(f"Found {len(metadata_files)} metadata files")
    
    migrated_count = 0
    skipped_count = 0
    error_count = 0
    
    for meta_path in metadata_files:
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            msg_id = metadata.get("id")
            if not msg_id:
                print(f"⚠️  No ID in {meta_path}")
                error_count += 1
                continue

            if not db.email_exists(msg_id):
                print(f"⏭️  Email {msg_id} not in database, skipping")
                skipped_count += 1
                continue
            
            attachments = metadata.get("attachments", [])
            if not attachments:
                continue
            
            existing_attachments = db.get_attachments_for_email(msg_id)
            existing_filenames = {att[0] for att in existing_attachments}
            
            for att in attachments:
                filename = att.get("filename")
                path = att.get("path")
                
                if not filename or not path:
                    continue
                
                if filename in existing_filenames:
                    continue
                
                if not os.path.exists(path):
                    print(f"  ⚠️  Attachment file not found: {path}")
                    continue
                
                db.insert_attachment(msg_id, filename, path)
                migrated_count += 1
                print(f"  ✅ Added: {msg_id} - {filename}")
        
        except Exception as e:
            print(f"❌ Error processing {meta_path}: {str(e)}")
            error_count += 1
    
    print(f"\n🎯 Migration complete:")
    print(f"   ✅ Migrated: {migrated_count} attachments")
    print(f"   ⏭️  Skipped: {skipped_count} emails (not in DB)")
    print(f"   ❌ Errors: {error_count}")
    
    cur = db.conn.cursor()
    cur.execute("SELECT COUNT(*) FROM attachments")
    total_attachments = cur.fetchone()[0]
    print(f"   📊 Total attachments in DB: {total_attachments}")
    
    db.close()

if __name__ == "__main__":
    print("Starting attachment migration...")
    migrate_attachments()

