from flask import Flask, request, jsonify
import threading
import fitz

from analyzer import analyze_pdf
from repository import (
    init_db, get_pdf_by_hash, insert_pdf, get_pdf_info, get_pdf_pages,
    get_page_analysis, get_page_groups, update_pdf_status
)
from utils import get_pdf_hash, base64_encode, base64_decode

app = Flask(__name__)
active_threads = {}  # Memastikan hanya satu thread per PDF

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

    # Hitung halaman PDF
    doc = fitz.open(stream=data, filetype="pdf")
    total_page = len(doc)

    # Insert ke DB
    pdf_id = insert_pdf(name, total_page, base64_encode(data), pdf_hash)

    # Jalankan thread hanya jika belum pernah dijalankan
    if pdf_id not in active_threads:
        def task():
            analyze_pdf(pdf_id, name, data)
            active_threads.pop(pdf_id, None)

        t = threading.Thread(target=task, daemon=True)
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
    progress = int((analyzed_count / int(total_page)) * 100) if total_page else 0

    result = []
    for page_id, page_num, page_name in pages:
        analysis = get_page_analysis(page_id)
        if not analysis:
            continue

        avg_similarity, page_valid = analysis
        
        groups = get_page_groups(page_id)
        group_data = [{
            "group_id": g[1],
            "similarity": g[2],
            "timestamp": g[3],
            "detail": g[4],
            "has_pole": g[5],
            "has_timestamp": g[6],
            "has_detail": g[7],
            "pole_name": g[8],
            "valid": g[9]
        } for g in groups]

        result.append({
            "page": page_num,
            "page_id": page_id,
            "page_name": page_name,
            "avg_similarity": avg_similarity,
            "page_valid": page_valid,
            "groups": group_data
        })

    sum_avg_similarity = round(sum(page["avg_similarity"] for page in result), 2) if result else 0
    return jsonify({
        "pdf_id": pdf_id,
        "pdf_name": pdf_name,
        "progress": progress,
        "sum_avg_similarity":sum_avg_similarity,
        "status": status,
        "message": "Berhasil Menganalisa PDF" if status == "completed" else "Masih diproses...",
        "result": result
    })


if __name__ == '__main__':
    init_db()
    app.run(host="0.0.0.0", port=5000)

