from pathlib import Path

from passport_service.objects import (
    DocMetadata,
    as_doc_metadata,
    parse_preprocessing_request,
)
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


def test_parse_preprocessing_request() -> None:
    # Given
    data_dir = TEST_DATA_DIR / "passports"
    docs = [
        str(data_dir / "some/path/with.ext"),
        str(data_dir / "some/other_path/with.another_ext"),
    ]

    # When
    doc_metadata = parse_preprocessing_request(docs, data_dir=data_dir)

    # Then
    expected = [
        DocMetadata(path=Path("some/path/with.ext"), extension=".ext"),
        DocMetadata(
            path=Path("some/other_path/with.another_ext"), extension=".another_ext"
        ),
    ]
    assert doc_metadata == expected
