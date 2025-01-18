"""Generate nanoid to be used in vector store as document id"""

from nanoid import generate


def gen_document_id() -> str:
    chars = "1234567890abcdefghijklmnopqrstuvwxyz"
    return generate(chars, size=10)
