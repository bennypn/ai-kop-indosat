# Import library Flask untuk web API, request untuk data masuk, jsonify untuk respons JSON
from flask import Flask, request, jsonify

# Import threading untuk menjalankan proses analisis PDF di background
import threading

# Import PyMuPDF (fitz) untuk membuka dan membaca file PDF
import fitz

# Import fungsi utama analisis PDF
from analyzer import analyze_pdf

# Import fungsi-fungsi interaksi database
from repository import (
    init_db,             # Inisialisasi database
    get_pdf_by_hash,     # Cek PDF apakah sudah pernah dianalisis (berdasarkan hash)
    insert_pdf,          # Menyimpan data PDF baru ke database
    get_pdf_info,        # Mengambil info metadata dari PDF tertentu
    get_pdf_pages,       # Mengambil data semua halaman PDF
    get_page_analysis,   # Mengambil hasil analisis per halaman
    get_page_groups,     # Mengambil grup yang ditemukan di suatu halaman
    update_pdf_status    # (Kemungkinan untuk update status, walau tidak dipakai di kode ini)
)

# Import utilitas untuk hash dan encoding base64
from utils import get_pdf_hash, base64_encode, base64_decode
from datetime import datetime

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Dictionary untuk menyimpan thread yang sedang aktif per PDF (agar tidak double processing)
active_threads = {}

# Endpoint /analyze untuk menerima file PDF dan menjalankan analisis
@app.route('/analyze', methods=['POST'])
def analyze():
    # Validasi: pastikan ada file yang diupload
    if 'file' not in request.files:
        return jsonify({"error": "No PDF uploaded"}), 400

    # Ambil file dari request
    file = request.files['file']
    
    # Validasi: pastikan filename tidak kosong
    if not file.filename:
        return jsonify({"error": "No filename provided"}), 400

    # Ambil nama file tanpa ekstensi
    name = file.filename.rsplit('.', 1)[0] if '.' in file.filename else file.filename

    # Baca isi file dalam bentuk byte
    data = file.read()

    # Validasi: file kosong
    if not data:
        return jsonify({"error": "Empty file"}), 400

    # Generate hash dari isi PDF untuk mengetahui apakah sudah pernah dianalisis
    pdf_hash = get_pdf_hash(data)
    existing = get_pdf_by_hash(pdf_hash)

    # Jika PDF sudah ada di DB
    if existing:
        pdf_id, status = existing
        if status != "completed":
            return jsonify({"pdf_id": pdf_id, "status": "in_process"})
        return jsonify({"pdf_id": pdf_id, "status": "completed"})

    # Hitung jumlah halaman PDF menggunakan PyMuPDF
    doc = fitz.open(stream=data, filetype="pdf")
    total_page = len(doc)

    # Insert PDF baru ke database
    pdf_id = insert_pdf(name, total_page, base64_encode(data), pdf_hash)

    # Jalankan thread hanya jika belum ada thread aktif untuk PDF tersebut
    if pdf_id not in active_threads:
        def task():
            analyze_pdf(pdf_id, name, data)  # Panggil fungsi analisis PDF
            active_threads.pop(pdf_id, None)  # Hapus thread dari daftar setelah selesai

        # Buat dan jalankan thread daemon
        t = threading.Thread(target=task, daemon=True)
        t.start()

        # Simpan referensi thread agar tidak dijalankan lagi
        active_threads[pdf_id] = t

    return jsonify({"pdf_id": pdf_id, "status": "started"})


# Endpoint /inquiry/<pdf_id> untuk mengambil hasil analisis PDF
@app.route('/inquiry/<int:pdf_id>', methods=['GET'])
def inquiry(pdf_id):
    # Ambil info PDF dari database
    info = get_pdf_info(pdf_id)
    if not info:
        return jsonify({"error": "PDF not found"}), 404

    # Unpack data dari database
    pdf_name, total_page, status, pdf_base64 = info

    # Ambil semua halaman dari PDF tersebut
    pages = get_pdf_pages(pdf_id)

    # Hitung jumlah halaman yang sudah dianalisis
    analyzed_count = len(pages)

    # Hitung progres dalam bentuk persen
    progress = int((analyzed_count / int(total_page)) * 100) if total_page else 0

    result = []  # List hasil analisis tiap hal
    sum_similarity = 0
    valid_pages = 0

    for page_id, page_num, page_name, page_url in pages:
        analysis = get_page_analysis(page_id)
        if not analysis:
            continue

        avg_similarity, page_valid, created_date = analysis

        # Hitung sum avg similarity
        if page_valid:
            sum_similarity += avg_similarity
            valid_pages += 1

        groups = get_page_groups(page_id)
        group_data = [dict(
            group_id=g[1],
            similarity=g[2],
            timestamp=g[3],
            detail=g[4],
            has_pole=g[5],
            has_timestamp=g[6],
            has_detail=g[7],
            pole_name=g[8],
            remark=g[9],  # Ambil remark dari grup
            valid=g[10],
            created_date=created_date,
            aging=(datetime.now() - created_date).days if isinstance(created_date, datetime) else None
        ) for g in groups]

        result.append({
            "page": page_num,
            "page_id": page_id,
            "page_name": page_name,
            "url": page_url,
            "avg_similarity": avg_similarity,
            "page_valid": page_valid,
            "groups": group_data,
            "created_date": created_date,
            "aging": (datetime.now() - created_date).days if isinstance(created_date, datetime) else None
        })

    sum_avg_similarity = round(sum_similarity, 2)

    return jsonify({
        "pdf_id": pdf_id,
        "pdf_name": pdf_name,
        "progress": progress,
        "status": status,
        "sum_avg_similarity": sum_avg_similarity,
        "message": "Berhasil Menganalisa PDF" if status == "completed" else "Masih diproses...",
        "valid_pages": valid_pages,
        "total_pages": total_page,
        "valid_percent": round((valid_pages / total_page) * 100, 2) if total_page else 0,
        "result": result
    })

if __name__ == '__main__':
    init_db()
    app.run(debug=False, host='0.0.0.0', port=5000)