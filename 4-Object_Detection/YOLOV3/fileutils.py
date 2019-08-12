#!/usr/bin/env python3
import cv2
from google.cloud import storage
import config as cfg
import boto3
import os
from util import image as image_util

# S3 Credentials
s3 = boto3.resource('s3')
s3_bucket = s3.Bucket(cfg.S3_BUCKET_NAME)

# Google Storage Credentials
gs_client = storage.Client()
gs_bucket = gs_client.get_bucket(cfg.GS_BUCKET_NAME)


def download_file_as_string(filepath: str):
    if filepath.startswith("gs://"):
        return download_gs_file_as_string(filepath)
    elif filepath.startswith("s3://"):
        return download_s3_file_as_string(filepath)
    elif filepath.startswith("/"):
        return download_local_file_as_string(filepath)


def download_gs_file_as_string(filepath):
    """download a gs file as a string"""
    filepath = filepath.replace("gs://{}/".format(cfg.GS_BUCKET_NAME), "")
    str_value = str(gs_bucket.blob(filepath).download_as_string())

    if str_value.startswith("b'"):
        str_value = str_value[2:]

    if str_value.endswith("'"):
        str_value = str_value[:-1]
    return str_value


def download_s3_file_as_string(filepath):
    """download a gs file as a string"""
    tempfilename = download_s3_file_as_string(filepath)
    with open(tempfilename, 'r') as f:
        return "\n".join(f.readlines())


def download_local_file_as_string(filepath):
    """download a gs file as a string"""
    with open(filepath, 'r') as f:
        return "\n".join(f.readlines())


def load_rgb_image(filepath: str):
    if filepath.startswith("gs://"):
        return load_image_from_gs(filepath)
    elif filepath.startswith("s3://"):
        return load_image_from_s3(filepath)
    elif filepath.startswith("/"):
        return load_image_from_local(filepath)


def load_image_from_gs(url):
    """
    :param url:
    :return:
    """
    filename = load_temp_file_from_gs(url)
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_image_from_s3(url):
    """
    :param url:
    :return:
    """
    filename = load_temp_file_from_s3(url)
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_image_from_local(url):
    """
    :param url:
    :return:
    """
    image = cv2.imread(url)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_temp_file_from_gs(url):
    """
    :param url:
    :return:
    """
    source_blob_name = "/".join(url.split("gs://")[1].split("/")[1:])
    blob = gs_bucket.blob(source_blob_name)

    import tempfile
    file_name = tempfile.mktemp()
    blob.download_to_filename(file_name)
    return file_name


def load_temp_file_from_s3(url):
    """
    :param url:
    :return:
    """
    import tempfile
    file_name = tempfile.mktemp()

    source_blob_name = "/".join(url.split("s3://")[1].split("/")[1:])
    s3_bucket.download_file(source_blob_name, file_name)
    return file_name


def if_file_exists(filepath):

    if filepath.startswith("gs://"):
        return if_gs_file_exists(filepath)
    elif filepath.startswith("s3://"):
        return if_s3_file_exists(filepath)
    elif filepath.startswith("/"):
        return if_local_file_exists(filepath)


def if_gs_file_exists(filepath):
    """If file exists"""
    blob = gs_bucket.blob(filepath.replace("gs://{}".format(cfg.GS_BUCKET_NAME), ""))
    return blob.exists()


def if_s3_file_exists(filepath):
    """If file exists"""
    source_blob_name = "/".join(filepath.split("s3://")[1].split("/")[1:])
    response = s3.list_objects_v2(
        Bucket=cfg.S3_BUCKET_NAME,
        Prefix=filepath,
    )
    for obj in response.get('Contents', []):
        if obj['Key'] == filepath:
            return True
    return False


def if_local_file_exists(filepath):
    """If file exists"""
    return os.path.exists(filepath)