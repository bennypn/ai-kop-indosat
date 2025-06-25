# Enhanced Flask App

from flask import Flask, request, jsonify
from ultralytics import YOLO
import fitz  # PyMuPDF
import easyocr
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from difflib import SequenceMatcher
import psycopg2
import re
import threading
import hashlib


app = Flask(__name__)
model = YOLO("tiang.pt")
reader = easyocr.Reader(['en'])

# DB Connection
conn = psycopg2.connect(
    host="localhost", port=5432,
    database="ai-core", user="postgres", password="postgres"
)
cursor = conn.cursor()

# DB Init

def init_db():
    cursor.execute("""
        CREATE SCHEMA IF NOT EXISTS kopindosat;

        CREATE TABLE IF NOT EXISTS kopindosat.pdfs (
            id SERIAL PRIMARY KEY,
            pdf_name TEXT,
            total_page INTEGER,
            url TEXT,
            description TEXT,
            status TEXT,
            base64 TEXT,
            hash TEXT UNIQUE
        );

        CREATE TABLE IF NOT EXISTS kopindosat.pdf_pages (
            id SERIAL PRIMARY KEY,
            pdf_id INTEGER REFERENCES kopindosat.pdfs(id) ON DELETE CASCADE,
            page INTEGER,
            page_name TEXT,
            url TEXT,
            description TEXT,
            status BOOLEAN,
            base64 TEXT
        );

        CREATE TABLE IF NOT EXISTS kopindosat.page_analysis (
            id SERIAL PRIMARY KEY,
            page_id INTEGER REFERENCES kopindosat.pdf_pages(id) ON DELETE CASCADE,
            avg_similarity FLOAT,
            page_valid BOOLEAN
        );

        CREATE TABLE IF NOT EXISTS kopindosat.page_analysis_group (
            id SERIAL PRIMARY KEY,
            anal_id INTEGER REFERENCES kopindosat.page_analysis(id) ON DELETE CASCADE,
            group_id INT,
            similarity FLOAT,
            timestamp TEXT,
            detail TEXT,
            has_pole BOOLEAN,
            has_timestamp BOOLEAN,
            has_detail BOOLEAN,
            pole_name TEXT,
            group_valid BOOLEAN
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_pdf_hash ON kopindosat.pdfs(hash);

    """)
    conn.commit()


# Helpers
def save_pdf_header_to_db(pdf_name, total_page, base64, pdf_hash, description="Uploaded", status=True):
    try:
        cursor.execute("""
            INSERT INTO kopindosat.pdfs (pdf_name, total_page, url, description, status, base64, hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (pdf_name, total_page, None, description, status, base64, pdf_hash))
        row = cursor.fetchone()
        if not row:
            raise Exception("❌ INSERT INTO kopindosat.pdfs gagal — tidak ada id yang dikembalikan")
        pdf_id = row[0]
        conn.commit()
        return pdf_id
    except Exception as e:
        conn.rollback()
        print(f"❌ Error saat insert PDF header: {e}")
        raise

def save_page_to_db(pdf_id, page, page_name, img_base64, description, status):
    cursor.execute("""
        INSERT INTO kopindosat.pdf_pages (pdf_id, page, page_name, url, description, status, base64)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """, (pdf_id, page, page_name, None, description, status, img_base64))
    page_id = cursor.fetchone()[0]
    conn.commit()
    return page_id

def ocr_text(image):
    return " ".join(reader.readtext(image, detail=0))

def compare_str(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def extract_pole_name(detail_text):
    match = re.search(r"Pole\s+Name\s+(.*?)\s+Pole\s+Hight", detail_text, re.IGNORECASE)
    return match.group(1).strip() if match else None

def check_base64_exists(img_base64):
    cursor.execute("SELECT id FROM kopindosat.pdf_pages WHERE base64 = %s", (img_base64,))
    found = cursor.fetchone()
    return found[0] if found else None

def check_pdf_exists(pdf_hash):
    cursor.execute("SELECT id FROM kopindosat.pdfs WHERE hash = %s", (pdf_hash,))
    result = cursor.fetchone()
    return result[0] if result else None

def get_pdf_status(pdf_id):
    cursor.execute("SELECT status FROM kopindosat.pdfs WHERE id = %s", (pdf_id,))
    result = cursor.fetchone()
    return result[0] if result else None

def update_pdf_status(pdf_id, status: str):
    cursor.execute("""
        UPDATE kopindosat.pdfs
        SET status = %s
        WHERE id = %s
    """, (status, pdf_id))
    conn.commit()

def get_pdf_hash(pdf_bytes):
    return hashlib.sha256(pdf_bytes).hexdigest()


def analyze_pdf_in_background(pdf_id, original_filename, pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num, page in enumerate(doc):
        page_name = f"{original_filename}_page{page_num + 1:02d}.png"
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_array = np.array(img)

            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            page_id = check_base64_exists(img_base64) or save_page_to_db(
                pdf_id, page_num + 1, page_name, img_base64,
                description="Berhasil convert ke gambar", status=True
            )

            yolo_result = model(img_array)[0]
            boxes = yolo_result.boxes.data
            labels = yolo_result.names

            detected, group_boxes = [], []
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box.tolist()
                label = labels[int(cls)]
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                detected.append({"label": label, "bbox": bbox})
                if label == "group":
                    group_boxes.append(bbox)

            groups, similarities = [], []
            group_counter = 1  # inisialisasi group ID global dalam PDF


            for gbox in group_boxes:
                gx1, gy1, gx2, gy2 = gbox
                has_pole = has_timestamp = has_detail = False
                timestamp_text = detail_text = ""

                for obj in detected:
                    lx1, ly1, lx2, ly2 = obj["bbox"]
                    if not (gx1 <= lx1 <= gx2 and gy1 <= ly1 <= gy2): continue
                    crop = img_array[ly1:ly2, lx1:lx2]
                    if obj["label"] == "pole": has_pole = True
                    elif obj["label"] == "timestamp":
                        has_timestamp = True
                        timestamp_text = ocr_text(crop)
                    elif obj["label"] == "detail":
                        has_detail = True
                        detail_text = ocr_text(crop)

                similarity = compare_str(timestamp_text, detail_text) if timestamp_text and detail_text else 0
                group_valid = has_pole and has_timestamp and has_detail and similarity >= 0.2
                similarities.append(similarity)

                # ✅ Ambil anal_id
                cursor.execute("SELECT id FROM kopindosat.page_analysis WHERE page_id = %s ORDER BY id DESC LIMIT 1", (page_id,))
                anal_row = cursor.fetchone()
                if not anal_row:
                    print(f"❌ No analysis found for page_id {page_id}")
                    continue

                anal_id = anal_row[0]

                # ✅ Insert ke table page_analysis_group
                cursor.execute("""
                    INSERT INTO kopindosat.page_analysis_group
                    (anal_id, group_id, similarity, timestamp, detail, has_pole, has_timestamp, has_detail, pole_name, group_valid)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    anal_id, group_counter, similarity, timestamp_text, detail_text,
                    has_pole, has_timestamp, has_detail,
                    extract_pole_name(detail_text), group_valid
                ))
                conn.commit()

                groups.append(group_valid)
                group_counter += 1  # 🔼 naikkan setelah 1 group selesai

            avg_sim = round(sum(similarities) / len(similarities), 2) if similarities else 0
            if avg_sim > 0.2:
                page_valid = True
            else:
                page_valid = False

            cursor.execute("""
                INSERT INTO kopindosat.page_analysis (page_id, avg_similarity, page_valid)
                VALUES (%s, %s, %s);
            """, (page_id, avg_sim, page_valid))
            conn.commit()

        except Exception as e:
            print(f"❌ Error analyzing PDF {pdf_id}: {e}")
            conn.rollback()  # Tambahkan ini
            update_pdf_status(pdf_id, "failed")
    update_pdf_status(pdf_id, "completed")
    print(f"✅ PDF {pdf_id} analysis completed.")


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No PDF uploaded"}), 400

    pdf_file = request.files['file']
    original_filename = pdf_file.filename.rsplit('.', 1)[0]
    pdf_bytes = pdf_file.read()
    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_hash = get_pdf_hash(pdf_bytes)  # 🔐 gunakan hash
    # Cek apakah file sudah dianalisis
    existing_pdf_id = check_pdf_exists(pdf_hash)

    if existing_pdf_id:
        status = get_pdf_status(existing_pdf_id)
        pdf_id = existing_pdf_id

        if status == "completed":
            return jsonify({
                "message": "PDF has already been analyzed.",
                "status": "completed",
                "pdf_id": existing_pdf_id,
                "pdf_name": original_filename
            })

        elif status == "in_process":
            # Bisa lanjutkan analisis yang belum selesai
            thread = threading.Thread(target=analyze_pdf_in_background, args=(existing_pdf_id, original_filename, pdf_bytes))
            thread.start()

            return jsonify({
                "message": "Previous analysis was incomplete, resuming.",
                "status": "resuming",
                "pdf_id": existing_pdf_id,
                "pdf_name": original_filename
            })
            
    # New PDF, insert
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_page = len(doc)

    pdf_id = save_pdf_header_to_db(
        pdf_name=original_filename,
        total_page=total_page,
        base64=pdf_base64,
        pdf_hash=pdf_hash,
        status="in_process"
    )


    thread = threading.Thread(target=analyze_pdf_in_background, args=(pdf_id, original_filename, pdf_bytes))
    thread.start()

    return jsonify({
        "message": "PDF accepted for analysis.",
        "status": "in_process",
        "pdf_id": pdf_id,
        "pdf_name": original_filename,
        "total_page": total_page
    })

@app.route('/inquiry/<int:pdf_id>', methods=['GET'])
def inquiry(pdf_id):
    # Ambil info pdf
    cursor.execute("SELECT pdf_name, total_page, status, base64 FROM kopindosat.pdfs WHERE id = %s", (pdf_id,))
    row = cursor.fetchone()

    if not row:
        return jsonify({"error": "PDF not found"}), 404

    pdf_name, total_page, status, pdf_base64 = row

    # Ambil page yang sudah dianalisis
    cursor.execute("SELECT id, page, page_name FROM kopindosat.pdf_pages WHERE pdf_id = %s", (pdf_id,))
    pages = cursor.fetchall()
    total_pages = int(total_page)
    analyzed_pages = len(pages)

    progress = int((analyzed_pages / total_pages) * 100) if total_pages else 0

    # Jalankan analisis ulang jika belum complete
    if status != "completed":
        print("⏳ Continuing background analysis...")
        t = threading.Thread(target=analyze_pdf_in_background, args=(pdf_id, pdf_name, base64.b64decode(pdf_base64)))
        t.start()

    # Ambil result yang sudah selesai
    result = []
    for page_id, page_num, page_name in pages:
        cursor.execute("""
            SELECT avg_similarity, page_valid FROM kopindosat.page_analysis WHERE page_id = %s
        """, (page_id,))
        anal = cursor.fetchone()
        if not anal:
            continue

        avg_similarity, page_valid = anal

        # Ambil group analysis
        cursor.execute("""
            SELECT g.similarity, g.timestamp, g.detail, g.has_pole, g.has_timestamp, 
                g.has_detail, g.pole_name, g.group_valid
            FROM kopindosat.page_analysis_group g
            JOIN kopindosat.page_analysis a ON g.anal_id = a.id
            WHERE a.page_id = %s
        """, (page_id,))

        groups = cursor.fetchall()
        group_results = []
        for g in groups:
            group_results.append({
                "similarity": g[0],
                "timestamp": g[1],
                "detail": g[2],
                "has_pole": g[3],
                "has_timestamp": g[4],
                "has_detail": g[5],
                "pole_name": g[6],
                "valid": g[7]
            })

        result.append({
            "page": page_num,
            "page_id": page_id,
            "page_name": page_name,
            "avg_similarity": avg_similarity,
            "page_valid": page_valid,
            "groups": group_results
        })

    return jsonify({
        "pdf_id": pdf_id,
        "pdf_name": pdf_name,
        "progress": progress,
        "status": status,
        "message": "Berhasil Menganalisa PDF" if status == "completed" else "Proses analisis masih berlangsung...",
        "result": result
    })

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
