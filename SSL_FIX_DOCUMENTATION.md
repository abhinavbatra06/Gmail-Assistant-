# SSL Certificate Fix for Docling/HuggingFace Downloads

## Problem
PDF and image attachments failed to process with SSL errors when Docling attempted to download AI models from HuggingFace:

```
SSLError: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1018)
```

## Root Cause
Outdated or misconfigured SSL certificates preventing secure connections to `huggingface.co` for model downloads.

## Solution

### Step 1: Install Required Packages
Install proper SSL certificate packages in your virtual environment:

```bash
# Activate your virtual environment first
# For Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Install SSL certificate packages
pip install --upgrade certifi python-certifi-win32 pyopenssl cryptography
pip install huggingface_hub==0.36.0
```

### Step 2: Update `src/docling_processor.py`
Add the following code at the **very beginning** of the file (before all other imports):

```python
# CRITICAL: Set environment variables BEFORE any imports
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'  # Fix Windows symlink permission error

# Fix SSL issues with HuggingFace using proper certificates
import ssl
import certifi
ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())
```

**Important Notes:**
- This MUST be placed before importing `docling` or any other modules
- The environment variables must be set before module imports to take effect
- This fixes both SSL certificate issues AND Windows symlink permission errors

### Step 3: Test the Fix
Test with a specific message that has PDF/image attachments:

```bash
# Navigate to project directory
cd Gmail-Assistant-

# Run docling processor for a specific email
python -m src.docling_processor --msg-id <message_id> --attachments
```

### Step 4: Process All Attachments
Once verified, process all emails with attachments:

```bash
python -m src.docling_processor --attachments
```

Or use the custom script:
```bash
python process_all_attachments.py
```

## What This Fix Does

1. **SSL Certificates**: Uses `certifi` to provide proper SSL certificate bundle for secure HTTPS connections
2. **Windows Symlinks**: Disables symlink creation in HuggingFace cache (requires admin rights on Windows)
3. **Model Downloads**: Allows Docling to successfully download required AI models (~172MB) from HuggingFace

## Verification

After the fix, you should see:
- ✅ HuggingFace models downloading successfully
- ✅ PDF text extraction working (e.g., 4,025 chars extracted from PDF)
- ✅ Image OCR processing working
- ✅ No SSL or symlink errors

## Files Modified

- `src/docling_processor.py` (lines 8-16)

## Git Commit Reference

```bash
git log --oneline | grep -i ssl
# cabe49e Fix: SSL certificate issue for HuggingFace model downloads
```

## Troubleshooting

### If downloads still fail:
1. Verify you're on a network that allows HTTPS to huggingface.co
2. Check if corporate firewall/proxy is blocking connections
3. Try on a different network (home WiFi, mobile hotspot)

### If symlink errors persist:
1. Run PowerShell as Administrator
2. Or enable Windows Developer Mode:
   - Settings → Update & Security → For Developers → Developer Mode

### If models are cached but still fail:
Delete the HuggingFace cache and re-download:
```bash
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\models--docling-project--docling-layout-heron"
```

## Related Issues

- Fixed in PR/Commit: `cabe49e`
- Merged with: Ankita's attachment processing improvements
- Date: December 2, 2025

## Success Metrics

Before fix:
- ❌ 6 PDF attachments failed
- ❌ 2 image attachments failed
- ✅ 1 DOCX attachment succeeded (no AI model needed)

After fix:
- ✅ All PDFs processing successfully
- ✅ All images processing successfully
- ✅ Text extraction working (~4,000+ chars per PDF)

## Dependencies Added

```txt
certifi>=2024.0.0
python-certifi-win32>=1.6.0
pyopenssl>=24.0.0
cryptography>=42.0.0
huggingface-hub==0.36.0
```
