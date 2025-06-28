import boto3
from botocore.client import Config
from datetime import datetime
import os
from config import CMS_CONFIG

# Inisialisasi koneksi S3
s3 = boto3.client(
    "s3",
    endpoint_url=CMS_CONFIG["CMS_ENDPOINT"],
    aws_access_key_id=CMS_CONFIG["CMS_ACCESS_KEY"],
    aws_secret_access_key=CMS_CONFIG["CMS_SECRET_KEY"],
    config=Config(signature_version="s3v4"),
    region_name="idn"  # default region
)

def upload_file(path: str, data: bytes) -> str:
    s3.put_object(
        Bucket=CMS_CONFIG["CMS_BUCKET"],
        Key=path,
        Body=data,
        ContentType='image/png',
        ACL='public-read'  # ⬅️ tambahkan ini agar file bisa diakses publik
    )
    return f"{CMS_CONFIG['CMS_ENDPOINT']}/{CMS_CONFIG['CMS_BUCKET']}/{path}"


