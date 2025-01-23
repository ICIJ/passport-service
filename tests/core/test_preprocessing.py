from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from icij_common.test_utils import fail_if_exception
from PIL import Image

from passport_service.core.pdf_conversion import GotenbergClient
from passport_service.core.preprocessing import preprocess_docs
from passport_service.objects import DocMetadata
from tests import TEST_DATA_DIR


# TODO: add a PDF in here when PDFs are supported
async def test_preprocess_docs(
    tmpdir: Path, test_gotenberg_client: GotenbergClient
) -> None:
    # Given
    tmpdir = Path(tmpdir)
    doc_root = TEST_DATA_DIR.joinpath("passports")
    docs = [
        DocMetadata(path=doc_root / "not_a_passport.jpg", extension=".jpg"),
        DocMetadata(path=doc_root / "passport.png", extension=".png"),
        DocMetadata(path=doc_root / "passport.pdf", extension=".pdf"),
        DocMetadata(path=doc_root / "passport.docx", extension=".docx"),
    ]
    with ProcessPoolExecutor(max_workers=2) as executor:
        # When
        reports = [
            report
            async for report in preprocess_docs(
                docs,
                tmpdir,
                executor,
                preprocessing_batch_size=1,
                gotenberg_client=test_gotenberg_client,
            )
        ]
        reports = sorted(reports, key=lambda r: r.doc_path)

    # Then
    docs = sorted(docs, key=lambda d: d.path)
    assert len(reports) == len(docs)
    for doc, report in zip(docs, reports):
        assert report.error is None
        assert report.doc_path == doc.path
        assert report.pages
        for page in report.pages:
            assert page.exists()
            with fail_if_exception(f"Failed to read valid png at {page}"):
                Image.open(page, formats=("png",))
