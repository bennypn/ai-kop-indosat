import psycopg2
from config import DB_CONFIG
from datetime import datetime

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

# --- DB Initialization ---
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
            hash TEXT UNIQUE,
            UNIQUE (pdf_name, total_page)
        );

        CREATE TABLE IF NOT EXISTS kopindosat.pdf_pages (
            id SERIAL PRIMARY KEY,
            pdf_id INTEGER REFERENCES kopindosat.pdfs(id) ON DELETE CASCADE,
            page INTEGER,
            page_name TEXT,
            url TEXT,
            description TEXT,
            status BOOLEAN,
            base64 TEXT,
            UNIQUE (pdf_id, page)
        );

        CREATE TABLE IF NOT EXISTS kopindosat.page_analysis (
            id SERIAL PRIMARY KEY,
            page_id INTEGER REFERENCES kopindosat.pdf_pages(id) ON DELETE CASCADE,
            avg_similarity FLOAT,
            page_valid BOOLEAN,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,                                      
            UNIQUE (page_id)
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
            remark TEXT,
            group_valid BOOLEAN,
            UNIQUE (anal_id, group_id)
        );
    """)
    conn.commit()

# --- PDF ---
def insert_pdf(name, total_page, base64_data, pdf_hash, description="Uploaded", status="in_process"):
    try:
        cursor.execute("""
            INSERT INTO kopindosat.pdfs
            (pdf_name, total_page, url, description, status, base64, hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (name, total_page, None, description, status, base64_data, pdf_hash))
        result = cursor.fetchone()
        conn.commit()
        return result[0]
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        cursor.execute("SELECT id FROM kopindosat.pdfs WHERE hash = %s", (pdf_hash,))
        existing = cursor.fetchone()
        return existing[0] if existing else None
    except Exception as e:
        conn.rollback()
        print(f"❌ Error insert PDF: {e}")
        raise

def get_pdf_by_hash(pdf_hash):
    cursor.execute("SELECT id, status FROM kopindosat.pdfs WHERE hash = %s", (pdf_hash,))
    return cursor.fetchone()

def get_pdf_info(pdf_id):
    cursor.execute("""
        SELECT pdf_name, total_page, status, base64
        FROM kopindosat.pdfs
        WHERE id = %s
    """, (pdf_id,))
    return cursor.fetchone()

def update_pdf_total_page(pdf_id, total_page):
    cursor.execute(
        "UPDATE kopindosat.pdfs SET total_page = %s WHERE id = %s",
        (total_page, pdf_id)
    )
    conn.commit()

def update_pdf_status(pdf_id, status):
    try:
        cursor.execute("UPDATE kopindosat.pdfs SET status = %s WHERE id = %s", (status, pdf_id))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"❌ Error update status PDF {pdf_id}: {e}")

# --- Pages ---
def save_page_to_db(pdf_id, page, page_name, img_base64, description, status, url):
    try:
        cursor.execute("""
            INSERT INTO kopindosat.pdf_pages
            (pdf_id, page, page_name, url, description, status, base64)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (pdf_id, page, page_name, url, description, status, img_base64))
        result = cursor.fetchone()
        conn.commit()
        return result[0]
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        cursor.execute("""
            SELECT id FROM kopindosat.pdf_pages
            WHERE pdf_id = %s AND page = %s
        """, (pdf_id, page))
        existing = cursor.fetchone()
        return existing[0] if existing else None
    except Exception as e:
        conn.rollback()
        print(f"❌ Error insert page {page_name}: {e}")
        raise

def get_pdf_pages(pdf_id):
    cursor.execute("""
        SELECT id, page, page_name, url
        FROM kopindosat.pdf_pages
        WHERE pdf_id = %s
        ORDER BY page
    """, (pdf_id,))
    return cursor.fetchall()

def check_base64_exists(img_base64):
    cursor.execute("SELECT id FROM kopindosat.pdf_pages WHERE base64 = %s", (img_base64,))
    result = cursor.fetchone()
    return result[0] if result else None

# --- Page Analysis ---
def insert_page_analysis(page_id, avg_similarity, page_valid):
    try:
        cursor.execute("""
            INSERT INTO kopindosat.page_analysis
            (page_id, avg_similarity, page_valid, created_date)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (page_id) DO NOTHING
        """, (page_id, avg_similarity, page_valid, datetime.now()))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"❌ Error insert page_analysis page_id {page_id}: {e}")

def get_page_analysis(page_id):
    cursor.execute("""
        SELECT avg_similarity, page_valid, created_date
        FROM kopindosat.page_analysis
        WHERE page_id = %s
        ORDER BY id DESC LIMIT 1
    """, (page_id,))
    return cursor.fetchone()

def get_analysis_id_by_page(page_id):
    cursor.execute("""
        SELECT id
        FROM kopindosat.page_analysis
        WHERE page_id = %s
        ORDER BY id DESC
    """, (page_id,))
    result = cursor.fetchone()
    return result[0] if result else None

# --- Group Analysis ---
def insert_group_analysis(
    anal_id, group_id, similarity, timestamp,
    detail, has_pole, has_timestamp, has_detail,
    pole_name, remark, group_valid
):
    try:
        cursor.execute("""
            INSERT INTO kopindosat.page_analysis_group
            (anal_id, group_id, similarity, timestamp, detail,
             has_pole, has_timestamp, has_detail, pole_name, remark, group_valid)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (anal_id, group_id) DO NOTHING
        """, (
            anal_id, group_id, similarity, timestamp, detail,
            has_pole, has_timestamp, has_detail, pole_name, remark, group_valid
        ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"❌ Error insert group_analysis anal_id {anal_id} group_id {group_id}: {e}")

def get_page_groups(page_id):
    cursor.execute("""
        SELECT
            g.id,              -- [0] ❌ Ini tidak dipakai di JSON
            g.group_id,        -- [1]
            g.similarity,      -- [2]
            g.timestamp,       -- [3]
            g.detail,          -- [4]
            g.has_pole,        -- [5]
            g.has_timestamp,   -- [6]
            g.has_detail,      -- [7]
            g.pole_name,       -- [8]
            g.remark,          -- [9]
            g.group_valid      -- [10]
        FROM kopindosat.page_analysis_group g
        JOIN kopindosat.page_analysis a ON g.anal_id = a.id
        WHERE a.page_id = %s
        ORDER BY g.group_id
    """, (page_id,))
    return cursor.fetchall()
