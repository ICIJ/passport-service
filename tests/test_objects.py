from pathlib import Path

from passport_service.objects import DocMetadata, as_doc_metadata
from tests import TEST_DATA_DIR


def test_as_doc_metadata() -> None:
    # Given
    data_dir = TEST_DATA_DIR / "passports"

    # When
    doc_metadata = as_doc_metadata(docs_dir=data_dir, data_dir=data_dir)

    # Then
    expected = [
        DocMetadata(path=Path("not_a_passport"), extension=".jpg"),
        DocMetadata(path=Path("not_a_passport.jpg"), extension=".jpg"),
        DocMetadata(path=Path("passport.docx"), extension=".docx"),
        DocMetadata(path=Path("passport.odt"), extension=".odt"),
        DocMetadata(path=Path("passport.pdf"), extension=".pdf"),
        DocMetadata(path=Path("passport.png"), extension=".png"),
    ]

    assert doc_metadata == expected
