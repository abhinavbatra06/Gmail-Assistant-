from src.docling_processor import DoclingProcessor
p = DoclingProcessor()
atts = p.process_attachments('19a5fb375bdbbaac', save=True)
print(f'\n Processed {len(atts)} attachments')
for a in atts:
    print(f'  - {a["filename"]}: {len(a.get("text", ""))} chars')
p.close()
