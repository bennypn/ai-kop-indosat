# Import library untuk PDF processing, image, dan threading
import fitz  # PyMuPDF untuk membaca dan render file PDF
import numpy as np  # Untuk manipulasi array gambar
from PIL import Image  # Library image processing
from io import BytesIO  # Untuk handling data dalam memory sebagai file
import threading  # Untuk menjalankan proses paralel/threading
from ultralytics import YOLO  # Library YOLOv8 untuk object detection

# Import fungsi utilitas
from utils import ocr_text, compare_str, extract_pole_name, base64_encode

# Import fungsi-fungsi database
from repository import (
    check_base64_exists,       # Mengecek apakah gambar halaman sudah ada di DB
    save_page_to_db,           # Menyimpan halaman PDF ke database
    update_pdf_status,         # Update status PDF ke 'completed'
    insert_group_analysis,     # Menyimpan analisis grup ke DB
    insert_page_analysis,      # Menyimpan analisis halaman ke DB
    get_analysis_id_by_page,   # Ambil ID analisis berdasarkan ID halaman
    get_page_groups            # Ambil data grup yang sudah dianalisis dari halaman
)

# Konfigurasi jumlah maksimal thread paralel yang boleh jalan bersamaan
from config import MAX_THREADS

from cms import upload_file  # Fungsi untuk upload file ke CMS

# ✅ Load model YOLO hasil training (format ONNX)
model = YOLO("train/weights/best.onnx")

# Batasi jumlah thread paralel yang berjalan
thread_queue = threading.BoundedSemaphore(MAX_THREADS)

# Fungsi utama untuk menganalisis PDF
def analyze_pdf(pdf_id, filename, pdf_bytes):
    with thread_queue:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for i in range(len(doc)):
            page = doc.load_page(i)
            page_name = f"{filename}_page{i + 1:02d}.png"

            # Convert PDF page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            img_array = np.array(img)

            # Simpan ke memory (BytesIO)
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            img_base64 = base64_encode(img_bytes)

            # Upload ke CMS dan dapatkan URL
            cms_path = f"/pdf/{pdf_id}/pdf-pages/{page_name}"
            img_url = upload_file(cms_path, img_bytes)

            # Simpan ke DB jika belum ada
            page_id = check_base64_exists(img_base64) or save_page_to_db(
                pdf_id=pdf_id,
                page=i + 1,
                page_name=page_name,
                img_base64=img_base64,
                description="Berhasil convert ke gambar",
                status=True,
                url=img_url  # 🆕 simpan URL hasil upload
            )

            # YOLO inference
            result = model(img_array)[0]
            boxes = result.boxes.data
            labels = result.names

            detected, group_boxes = [], []
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box.tolist()
                label = labels[int(cls)]
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                detected.append({"label": label, "bbox": bbox})
                if label == "group":
                    group_boxes.append(bbox)

            similarities = []
            group_counter = 1
            avg_sim = 0.0
            page_valid = False

            if group_boxes:
                for gbox in group_boxes:
                    gx1, gy1, gx2, gy2 = gbox
                    has_pole = has_timestamp = has_detail = False
                    timestamp_text = detail_text = ""

                    for obj in detected:
                        lx1, ly1, lx2, ly2 = obj["bbox"]
                        if not (gx1 <= lx1 <= gx2 and gy1 <= ly1 <= gy2):
                            continue
                        crop = img_array[ly1:ly2, lx1:lx2]

                        if obj["label"] == "pole":
                            has_pole = True
                        elif obj["label"] == "timestamp":
                            has_timestamp = True
                            timestamp_text = ocr_text(crop)
                        elif obj["label"] == "detail":
                            has_detail = True
                            detail_text = ocr_text(crop)

                    similarity = compare_str(timestamp_text, detail_text) if timestamp_text and detail_text else 0
                    group_valid = has_pole and has_timestamp and has_detail and similarity >= 0.2
                    similarities.append(similarity)

                avg_sim = round(sum(similarities) / len(similarities), 2) if similarities else 0
                page_valid = avg_sim > 0.2

            insert_page_analysis(page_id, avg_sim, page_valid)
            anal_id = get_analysis_id_by_page(page_id)

            if anal_id:
                existing_groups = get_page_groups(page_id)
                existing_group_ids = {g[1] for g in existing_groups}

                for gbox in group_boxes:
                    gx1, gy1, gx2, gy2 = gbox
                    has_pole = has_timestamp = has_detail = False
                    timestamp_text = detail_text = ""

                    for obj in detected:
                        lx1, ly1, lx2, ly2 = obj["bbox"]
                        if not (gx1 <= lx1 <= gx2 and gy1 <= ly1 <= gy2):
                            continue
                        crop = img_array[ly1:ly2, lx1:lx2]

                        if obj["label"] == "pole":
                            has_pole = True
                        elif obj["label"] == "timestamp":
                            has_timestamp = True
                            timestamp_text = ocr_text(crop)
                        elif obj["label"] == "detail":
                            has_detail = True
                            detail_text = ocr_text(crop)

                    similarity = compare_str(timestamp_text, detail_text) if timestamp_text and detail_text else 0
                    group_valid = has_pole and has_timestamp and has_detail and similarity >= 0.2

                    if group_counter not in existing_group_ids:
                        insert_group_analysis(
                            anal_id=anal_id,
                            group_id=group_counter,
                            similarity=similarity,
                            timestamp=timestamp_text,
                            detail=detail_text,
                            has_pole=has_pole,
                            has_timestamp=has_timestamp,
                            has_detail=has_detail,
                            pole_name=extract_pole_name(detail_text),
                            group_valid=group_valid
                        )
                    group_counter += 1

        update_pdf_status(pdf_id, "completed")
        print(f"✅ PDF {pdf_id} analysis completed.")