from src.docling_processor import DoclingProcessor

print('Testing attachment processing with SSL fix...')
p = DoclingProcessor()
atts = p.process_attachments('19a5fb375bdbbaac', save=True)

print(f'\n=== RESULTS ===')
print(f'Processed {len(atts)} attachments')
for a in atts:
    chars = len(a.get('text', ''))
    has_error = 'error' in a.get('metadata', {})
    status = ' ERROR' if has_error else ' SUCCESS'
    print(f'{status} {a["filename"]}: {chars} chars')
    if has_error:
        print(f'  Error: {a["metadata"]["error"][:100]}...')

p.close()
