try:
    from .object_detection import detect_passports
except ModuleNotFoundError:
    detect_passports = None
try:
    from .pdf_conversion import GotenbergClient
except ModuleNotFoundError:
    GotenbergClient = None
try:
    from .preprocessing import (
        async_process_pdf,
        get_pil_supported_extensions,
        preprocess_docs,
        process_image,
        process_pdf,
        should_convert_to_pdf,
    )
except ModuleNotFoundError:
    async_process_pdf = None
    get_pil_supported_extensions = None
    preprocess_docs = None
    process_image = None
    process_pdf = None
    should_convert_to_pdf = None

try:
    from .mrz import read_passport_file_mrz
except ModuleNotFoundError:
    read_passport_file_mrz = None

__all__ = [
    "GotenbergClient",
    "get_pil_supported_extensions",
    "preprocess_docs",
    "async_process_pdf",
    "process_image",
    "process_pdf",
    "should_convert_to_pdf",
    "detect_passports",
    "read_passport_file_mrz",
]
