from flask import Flask, request, jsonify
import threading
import fitz  # Import the missing module
from analyzer import analyze_pdf
from repository import (
    init_db, get_pdf_by_hash, insert_pdf, get_pdf_info, get_pdf_pages,
    get_page_analysis, get_page_groups, update_pdf_status
)
from utils import get_pdf_hash, base64_encode, base64_decode

# app.py (global)
app = Flask(__name__)
active_threads = {}  # key = pdf_id, value = threading.Thread


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No PDF uploaded"}), 400
    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "No filename provided"}), 400

    name = file.filename.rsplit('.', 1)[0] if '.' in file.filename else file.filename
    data = file.read()
    if not data:
        return jsonify({"error": "Empty file"}), 400

    pdf_hash = get_pdf_hash(data)
    existing = get_pdf_by_hash(pdf_hash)
    if existing:
        pdf_id, status = existing
        if status != "completed":
            return jsonify({"pdf_id": pdf_id, "status": "in_process"})
        return jsonify({"pdf_id": pdf_id, "status": "completed"})

    # Hitung halaman
    doc = fitz.open(stream=data, filetype="pdf")
    total_page = len(doc)

    pdf_id = insert_pdf(name, total_page, base64_encode(data), pdf_hash)

    # Jangan start thread kalau sudah ada untuk pdf ini
    if pdf_id not in active_threads:
        def task():
            analyze_pdf(pdf_id, name, data)
            del active_threads[pdf_id]

        t = threading.Thread(target=task)
        t.start()
        active_threads[pdf_id] = t

    return jsonify({"pdf_id": pdf_id, "status": "started"})

@app.route('/inquiry/<int:pdf_id>', methods=['GET'])
def inquiry(pdf_id):
    info = get_pdf_info(pdf_id)
    if not info:
        return jsonify({"error": "PDF not found"}), 404

    pdf_name, total_page, status, pdf_base64 = info
    pages = get_pdf_pages(pdf_id)
    analyzed_count = len(pages)
    progress = int((analyzed_count / total_page) * 100) if total_page else 0

    # ❌ Jangan trigger ulang analisis jika status masih in_process
    # ❌ Jangan jalankan thread baru di sini
    result = []
    for page_id, page_num, page_name in pages:
        analysis = get_page_analysis(page_id)
        if not analysis:
            continue
        avg_similarity, page_valid = analysis

        groups = get_page_groups(page_id)
        group_data = [dict(
            similarity=g[0], timestamp=g[1], detail=g[2],
            has_pole=g[3], has_timestamp=g[4], has_detail=g[5],
            pole_name=g[6], valid=g[7]
        ) for g in groups]

        result.append({
            "page": page_num,
            "page_id": page_id,
            "page_name": page_name,
            "avg_similarity": avg_similarity,
            "page_valid": page_valid,
            "groups": group_data
        })

    return jsonify({
        "pdf_id": pdf_id,
        "pdf_name": pdf_name,
        "progress": progress,
        "status": status,
        "message": "Berhasil Menganalisa PDF" if status == "completed" else "Masih diproses...",
        "result": result
    })

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
