"""
Reprocess all emails to extract PDF and image attachments that previously failed
"""
import os
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

from src.docling_processor import DoclingProcessor

print("🚀 Starting attachment reprocessing...")
print("This will process all emails and extract text from PDF and image attachments\n")

processor = DoclingProcessor()

# Get all email IDs that have attachments
cur = processor.db.conn.cursor()
cur.execute("""
    SELECT DISTINCT e.id, e.subject 
    FROM emails e
    INNER JOIN attachments a ON e.id = a.message_id
    WHERE e.status = 'processed'
""")

emails_with_attachments = cur.fetchall()
print(f"Found {len(emails_with_attachments)} emails with attachments\n")

success_count = 0
error_count = 0
total_chars = 0

for msg_id, subject in emails_with_attachments:
    print(f"\n📧 Processing: {subject[:60]}...")
    try:
        atts = processor.process_attachments(msg_id, save=True)
        for att in atts:
            chars = len(att.get('text', ''))
            total_chars += chars
            has_error = 'error' in att.get('metadata', {})
            if has_error:
                error_count += 1
                print(f"  ❌ {att['filename']}: ERROR")
            else:
                success_count += 1
                print(f"  ✅ {att['filename']}: {chars} chars")
    except Exception as e:
        error_count += 1
        print(f"  ❌ Error: {str(e)[:100]}")

print(f"\n{'='*60}")
print(f"✅ Successfully processed: {success_count} attachments")
print(f"❌ Failed: {error_count} attachments")
print(f"📊 Total text extracted: {total_chars:,} characters")
print(f"{'='*60}")

processor.close()
