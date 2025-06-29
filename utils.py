# === utils.py ===
import base64
import hashlib
import re
from difflib import SequenceMatcher
import easyocr

reader = easyocr.Reader(['en'])
reader = easyocr.Reader(['en'])

def get_pdf_hash(pdf_bytes):
    return hashlib.sha256(pdf_bytes).hexdigest()

def base64_encode(data):
    return base64.b64encode(data).decode("utf-8")

def base64_decode(data):
    return base64.b64decode(data)

def ocr_text(image):
    return " ".join([x for x in reader.readtext(image, detail=0) if isinstance(x, str)])

def compare_str(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def extract_pole_name(detail_text):
    match = re.search(r"Pole\s+Name\s+(.*?)\s+Pole\s+Hight", detail_text, re.IGNORECASE)
    return match.group(1).strip() if match else None

def extract_remark(detail_text: str) -> str:
    match = re.search(r'Remark\s+(.*)', detail_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ''
