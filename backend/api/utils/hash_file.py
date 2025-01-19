import hashlib


def get_file_hash(file_url: str):
    return hashlib.md5(open(file_url, "rb").read()).hexdigest()
