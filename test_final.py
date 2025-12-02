import os
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

from src.docling_processor import DoclingProcessor
p = DoclingProcessor()
atts = p.process_attachments('19a5fb375bdbbaac', save=True)
print(f'\nProcessed: {len(atts)} attachments')
for a in atts:
    chars = len(a.get('text', ''))
    print(f'{a["filename"]}: {chars} chars')
p.close()
