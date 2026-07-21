try:
    from .object_detection import detect_passports
except ModuleNotFoundError:
    detect_passports = None
try:
    from .pdf_conversion import GotenbergClient
except ModuleNotFoundError:
    GotenbergClient = None
try:
    from .preprocessing import preprocess_docs
except ModuleNotFoundError:
    preprocess_docs = None

try:
    from .mrz import read_passport_file_mrz
except ModuleNotFoundError:
    preprocess_docs = None

__all__ = [
    "GotenbergClient",
    "detect_passports",
    "preprocess_docs",
    "read_passport_file_mrz",
]
