import fitz
import numpy as np
from PIL import Image
from io import BytesIO
import threading
from ultralytics import YOLO

from utils import ocr_text, compare_str, extract_pole_name, base64_encode
from repository import (
    check_base64_exists, save_page_to_db, update_pdf_status,
    insert_group_analysis, insert_page_analysis, get_analysis_id_by_page,
    get_page_groups
)
from config import MAX_THREADS

model = YOLO("tiang.pt")
thread_queue = threading.BoundedSemaphore(MAX_THREADS)

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

            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img_base64 = base64_encode(buffer.getvalue())

            # Simpan page jika belum ada
            page_id = check_base64_exists(img_base64) or save_page_to_db(
                pdf_id, i + 1, page_name, img_base64,
                description="Berhasil convert ke gambar", status=True
            )

            # Object detection dengan YOLO
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

            # 💾 Insert page_analysis lebih dulu agar anal_id tersedia
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

            # Insert group jika belum ada
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
                    else:
                        print(f"⏭️ Group {group_counter} already exists in DB, skipping.")

                    group_counter += 1

        update_pdf_status(pdf_id, "completed")
        print(f"✅ PDF {pdf_id} analysis completed.")
