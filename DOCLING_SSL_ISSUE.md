# Docling Attachment Processing SSL Issue

## Issue Summary
PDF and image attachments are failing to process due to SSL connection errors when Docling tries to download AI models from HuggingFace.

## Current Status
- ✅ **DOCX files work perfectly** (e.g., "251001 Lepercq internship.docx" extracted 3,500 chars successfully)
- ❌ **PDF files fail** (6 PDFs all failed with SSL errors)
- ❌ **Image files fail** (JPG, PNG files fail with SSL errors)
- ✅ **Email body processing works** (all 95 emails processed successfully)

## Failed Attachments
1. `IPPE_AI_Investment_Tech_Internship_Job_Description v2.pdf`
2. `2026_BME-CAIR_Summer Program Flyer.pdf`
3. `2026 BME-CAIR Summer Program Brochure.pdf`
4. `2026 BME-CAIR Research Project Descriptions.pdf`
5. `Genhack-Challenge.jpg`
6. `image.png`

## Root Cause
**SSL Protocol Error** when connecting to `huggingface.co`:
```
SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1018)'))
```

This happens when Docling's `DocumentConverter` tries to:
1. Download the `docling-layout-heron` model from HuggingFace
2. Verify/update the cached model files
3. Download additional dependencies (`.gitattributes` file)

## Environment Context
- **Network**: Google Drive path suggests possible network restrictions (corporate/school network)
- **Cache**: HuggingFace models ARE cached locally at `C:\Users\abhin\.cache\huggingface\hub\models--docling-project--docling-layout-heron`
- **Issue**: Even with cached models, Docling tries to verify/update them online and fails

## Technical Details

### Error Chain
1. Docling initializes `DocumentConverter()` 
2. For PDFs/images, it needs layout detection models
3. HuggingFace Hub library tries to verify model integrity
4. HTTPS connection to `huggingface.co` starts
5. SSL handshake fails mid-protocol
6. Processing aborts with error

### Why It's Not Simple
- Setting `HF_HUB_OFFLINE=1` causes "outgoing traffic disabled" error
- SSL verification bypass didn't work (library already imported)
- Models are cached but still needs online verification
- Environment variables need to be set BEFORE library imports

## Attempted Solutions (Did Not Work)
1. ❌ SSL verification bypass after imports
2. ❌ Offline mode (causes different error)
3. ❌ Environment variable changes after module load

## Recommended Solutions

### Solution 1: Pre-load Models (BEST)
Download models when network is available, then force offline mode properly:

```bash
# One-time: Download models on a different network
python -c "from huggingface_hub import snapshot_download; snapshot_download('docling-project/docling-layout-heron')"

# Then in docling_processor.py, BEFORE any imports:
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
```

### Solution 2: Use Different Network
- Try on home network instead of school/work network
- Use mobile hotspot
- Use VPN that doesn't interfere with SSL

### Solution 3: Fallback to Simple Extractors
Add try/except in `process_attachments()` method to use simpler libraries when Docling fails:

```python
# For PDFs
import pdfplumber
import PyPDF2

# For images  
import pytesseract
from PIL import Image
```

### Solution 4: Proxy Configuration
If behind corporate proxy:

```python
import os
os.environ['HTTP_PROXY'] = 'http://proxy:port'
os.environ['HTTPS_PROXY'] = 'http://proxy:port'
```

## Testing Command
After implementing fix, test with:

```bash
python -c "from src.docling_processor import DoclingProcessor; p = DoclingProcessor(); atts = p.process_attachments('19a5fb375bdbbaac', save=True); print(f'Processed {len(atts)} attachments'); p.close()"
```

Should process the PDF and extract text content (not return 0 chars).

## File Locations
- Processed emails: `data/docling/*.json`
- Failed attachment JSONs: `data/docling/*_att_*.json` (check for "error" in metadata)
- Attachment source files: `data/attachments/`
- Code: `src/docling_processor.py` (line ~200+ for `process_attachments()` method)

## Priority
**Medium-High** - Email bodies work fine, but attachment content (especially PDFs with internship descriptions, program details) is valuable for Q&A system.

## Update: SSL Workaround Did NOT Work

**Tested**: Setting `CURL_CA_BUNDLE=''`, `REQUESTS_CA_BUNDLE=''`, and `ssl._create_unverified_context` 
**Result**: Still fails with same SSL error - network is blocking at lower level

This confirms it's a **network/firewall issue**, not fixable with code changes.

## Next Steps (IN ORDER OF PRIORITY)

### REQUIRED: Change Network
**YOU MUST try on a different network to use Docling**:
1. ✅ **Home WiFi** (not school/work network)  
2. ✅ **Mobile hotspot** from your phone
3. ✅ **Coffee shop / public WiFi**
4. ✅ **Friend's network**

Once on a working network, the models will download and cache. Then you can go back to your current network.

### If Network Change Not Possible
Implement fallback to simple PDF extractors (PyPDF2, pdfplumber) - see Solution 3 in document above.
