from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, StreamingHttpResponse
from django.core.files.storage import default_storage
from django.conf import settings
import os
import json
import uuid
import requests
import docx
import pandas as pd
from PIL import Image, ImageFilter
import fitz  # PyMuPDF for PDF rendering
import io
import pytesseract

# ===============================
# Tesseract OCR setup
# ===============================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"  # Ensure ben.traineddata is here

def chat_ui(request):
    return render(request, 'chatbox/index.html')

def preprocess_image(image_path):
    """Enhance image quality for better OCR accuracy."""
    img = Image.open(image_path).convert('L')  # grayscale
    if os.path.getsize(image_path) > 500_000:  # sharpen only big images
        img = img.filter(ImageFilter.SHARPEN)
    return img

def extract_large_pdf(file_path, chunk_size=3000):
    """
    Efficiently extract text from scanned PDFs (Nikosh font) page by page.
    Uses OCR on pages that have no extractable text.
    """
    try:
        doc = fitz.open(file_path)
        buffer_text = ""

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()  # Try normal text extraction

            # Fallback to OCR for scanned / non-Unicode fonts
            if not text.strip():
                pix = page.get_pixmap(dpi=300)  # high-res image
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                text = pytesseract.image_to_string(img, lang="ben+eng", config="--psm 6")

            buffer_text += f"\n--- Page {page_num+1} ---\n{text}"

            # Yield chunks if buffer too big
            if len(buffer_text) >= chunk_size:
                yield buffer_text
                buffer_text = ""

        if buffer_text:
            yield buffer_text

    except Exception as e:
        yield f"[ERROR_EXTRACTING_PDF: {str(e)}]"

def _extract_text_from_file(file_path, chunk_size=3000):
    """
    Extracts text from multiple file formats: txt, pdf, docx, images, excel.
    Splits large outputs into chunks.
    """
    ext = os.path.splitext(file_path)[1].lower()
    full_text = ""

    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                full_text = f.read()

        elif ext == ".pdf":
            text_chunks = []
            for chunk in extract_large_pdf(file_path, chunk_size=chunk_size):
                text_chunks.append(chunk)
            full_text = "\n---CHUNK_BREAK---\n".join(text_chunks)

        elif ext == ".docx":
            doc = docx.Document(file_path)
            full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path, sheet_name=None)
            for sheet_name, sheet_df in df.items():
                full_text += f"\n--- Sheet: {sheet_name} ---\n"
                full_text += sheet_df.to_string(index=False)

        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            img = preprocess_image(file_path)
            full_text = pytesseract.image_to_string(img, lang="ben+eng", config="--psm 6")

        else:
            full_text = f"[UNSUPPORTED FILE TYPE: {ext}]"

    except Exception as e:
        return f"[ERROR_EXTRACTING_FILE: {str(e)}]"

    # Chunk large text
    chunks = []
    text_len = len(full_text)
    for i in range(0, text_len, chunk_size):
        chunks.append(full_text[i:i + chunk_size])

    return "\n---CHUNK_BREAK---\n".join(chunks)

def _user_requested_export(message_text):
    """Check if user explicitly asked for a downloadable file."""
    if not message_text:
        return False
    m = message_text.lower()
    keywords = [
        "download", "export", "json", "excel", "xlsx",
        "csv", "file", "download json", "download excel", "save as"
    ]
    return any(k in m for k in keywords)

@csrf_exempt
def send_to_ollama(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method."}, status=405)

    user_msg = request.POST.get("message", "").strip()
    uploaded_files = request.FILES.getlist("files")

    extracted_per_file = {}

    # Extract text from files
    for uploaded_file in uploaded_files:
        saved = default_storage.save(uploaded_file.name, uploaded_file)
        abs_path = default_storage.path(saved)
        text = _extract_text_from_file(abs_path)
        extracted_per_file[uploaded_file.name] = text
        try:
            os.remove(abs_path)
        except Exception:
            pass

    wants_export = _user_requested_export(user_msg)
    json_url = None
    json_filename = None

    # Create JSON if requested
    if wants_export and extracted_per_file:
        json_filename = f"extracted_{uuid.uuid4().hex}.json"
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        json_path = os.path.join(settings.MEDIA_ROOT, json_filename)
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(extracted_per_file, jf, ensure_ascii=False, indent=2)
        json_url = f"/media/{json_filename}"

    # Build prompt for the model
    if not uploaded_files:
        prompt = user_msg or "Hello"
    else:
        prompt = "You are an assistant. The user uploaded file(s) whose extracted text is below.\n\n"
        for fname, txt in extracted_per_file.items():
            prompt += f"---FILE_START---\nFilename: {fname}\nContent:\n{txt}\n---FILE_END---\n\n"
        prompt += "User's question or instruction:\n" + (user_msg or "Please summarize the files.")

    # Streaming response from Ollama
    def stream_response():
        try:
            with requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "deepseek-r1:7b", "prompt": prompt, "stream": True},
                stream=True,
                timeout=1800
            ) as r:
                r.encoding = 'utf-8'
                for line in r.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode("utf-8"))
                            if "response" in data:
                                yield data["response"]
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"[ERROR_CALLING_MODEL: {str(e)}]"

    # If streaming is requested
    if request.headers.get("X-Stream", "").lower() == "true":
        return StreamingHttpResponse(stream_response(), content_type="text/plain; charset=utf-8")

    # Non-streaming fallback
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "deepseek-r1:7b", "prompt": prompt, "stream": False},
            timeout=1800
        )
        resp.encoding = 'utf-8'
        model_out = resp.json().get("response", "")
    except Exception as e:
        model_out = f"[ERROR_CALLING_MODEL: {str(e)}]"

    return JsonResponse({
        "response": model_out,
        "json_url": json_url,
        "json_filename": json_filename
    })
