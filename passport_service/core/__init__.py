try:
    from passport_service.core.object_detection import detect_passports
except ModuleNotFoundError:
    detect_passports = None
try:
    from passport_service.core.pdf_conversion import GotenbergClient
except ModuleNotFoundError:
    GotenbergClient = None
try:
    from passport_service.core.preprocessing import preprocess_docs
except ModuleNotFoundError:
    preprocess_docs = None

try:
    from passport_service.core.mrz import read_passport_file_mrz
except ModuleNotFoundError:
    preprocess_docs = None

__all__ = [
    "GotenbergClient",
    "detect_passports",
    "preprocess_docs",
    "read_passport_file_mrz",
]
