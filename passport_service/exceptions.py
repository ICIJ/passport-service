from abc import ABC
from pathlib import Path


class UnsupportedDocExtension(ValueError):
    def __init__(self, ext: str, supported: list[str]):
        msg = (
            f"Unsupported document extension: {ext}.\nSupported extensions: {supported}"
        )
        super().__init__(msg)


class InvalidDocument(RuntimeError, ABC):
    doc_label: str

    def __init__(self, path: Path, reason: str | None = None):
        msg = f"Invalid {self.doc_label}: {path}."
        if reason is not None:
            msg = f"{msg}. {reason}"
        super().__init__(msg)


class InvalidPDF(InvalidDocument):
    doc_label: str = "PDF"


class InvalidImage(InvalidDocument):
    doc_label: str = "image"
