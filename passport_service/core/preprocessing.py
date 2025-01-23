import hashlib
import logging
import os
from collections.abc import AsyncGenerator, Generator, Iterable
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import lru_cache, partial
from pathlib import Path

import pymupdf
from icij_common.pydantic_utils import safe_copy
from icij_worker.typing_ import RateProgress, RawProgress
from icij_worker.utils.progress import to_raw_progress, to_scaled_progress
from PIL import Image, UnidentifiedImageError
from pymupdf import EmptyFileError, FileDataError

from passport_service.utils import run_with_concurrency

from ..constants import COLOR_LUT, PDF_EXT, PIL_PNG, PNG_EXT, Colorspace
from ..exceptions import InvalidImage, InvalidPDF, UnsupportedDocExtension
from ..objects import DocMetadata, Error, ProcessingReport
from .pdf_conversion import GotenbergClient, should_convert_to_pdf

DEFAULT_DOC_PROCESSING_BATCH_SIZE = 10

logger = logging.getLogger(__name__)


def get_pil_supported_extensions() -> list[str]:
    exts = Image.registered_extensions()
    return [ex for ex, f in exts.items() if f in Image.OPEN]


PIL_SUPPORTED_EXTENSIONS = get_pil_supported_extensions()
SUPPORTED_DOC_EXTS = set([PDF_EXT] + PIL_SUPPORTED_EXTENSIONS)
SUPPORTED_DOC_EXTS_LIST = sorted(SUPPORTED_DOC_EXTS)

REPORTED_ERRORS = (UnsupportedDocExtension, InvalidPDF)


def _scale_pdf_progress(
    progress: RateProgress, *, n_pdfs: int, pdf_processing_end: float
) -> RawProgress:
    progress = to_scaled_progress(progress, end=pdf_processing_end)
    return to_raw_progress(progress, max_progress=n_pdfs)


async def preprocess_docs(
    docs_metadata: list[DocMetadata],
    output_dir: Path,
    executor: ProcessPoolExecutor,
    gotenberg_client: GotenbergClient,
    preprocessing_batch_size: int,
    *,
    pdf_conversion_concurrency: int = 2,
    # TODO: use an enum here ?
    colorspace: Colorspace = Colorspace.RGB,
    progress: RateProgress | None = None,
) -> Generator[ProcessingReport, None, None]:
    n_processes = executor._max_workers
    # TODO: remove this when a proper PDF service is running
    docs_metadata = deepcopy(docs_metadata)
    # TODO: input data should be deduped based on doc.path to avoid processing the same
    #  file several times
    n_docs = len(docs_metadata)
    logger.debug("Pre processing %s docs...", n_docs)
    to_convert_to_pdf = [
        (i, meta)
        for i, meta in enumerate(docs_metadata)
        if should_convert_to_pdf(meta.extension)
    ]
    # If all docs are PDFs, let's say the conversion takes about 80% of the time
    base_pdf_processing_end = 0.8 if to_convert_to_pdf else 0.0
    pdf_processing_end = base_pdf_processing_end * base_pdf_processing_end
    if to_convert_to_pdf:
        pdf_progress = None
        if progress is not None:
            pdf_progress = _scale_pdf_progress(
                progress,
                n_pdfs=len(to_convert_to_pdf),
                pdf_processing_end=pdf_processing_end,
            )
        logger.debug("Convert %s docs to PDF...", len(to_convert_to_pdf))
        n_pdf = 0
        pdf_it = _convert_to_pdf(
            to_convert_to_pdf,
            client=gotenberg_client,
            max_concurrency=pdf_conversion_concurrency,
        )
        async for doc_i, doc_path, pdf_bytes in pdf_it:
            pdf_path = output_dir / make_pdf_filename(doc_path)
            pdf_path.write_bytes(pdf_bytes)
            docs_metadata[doc_i] = safe_copy(
                docs_metadata[doc_i], update={"pdf_path": pdf_path}
            )
            n_pdf += 1
            if pdf_progress is not None:
                await pdf_progress(n_pdf)
    if progress is not None:
        progress = to_scaled_progress(progress, start=pdf_processing_end, end=1.0)
        progress = to_raw_progress(progress, len(docs_metadata))
    chunk_size = (
        1
        if n_docs < n_processes * preprocessing_batch_size
        else preprocessing_batch_size
    )
    process_doc_fn = partial(
        preprocess_doc, output_dir=output_dir, colorspace=colorspace
    )
    n_processed = 0
    update_progress_every = 5
    for report in executor.map(process_doc_fn, docs_metadata, chunksize=chunk_size):
        yield report
        n_processed += 1
        if progress and not n_processed % update_progress_every:
            progress(n_processed)
    if progress and n_processed % update_progress_every:
        progress(n_processed)


def preprocess_doc(
    meta: DocMetadata, output_dir: Path, colorspace: Colorspace
) -> ProcessingReport:
    try:
        if meta.extension == PDF_EXT:  # Original pdf
            processing_fn = partial(
                process_pdf, pdf_path=meta.path, colorspace=colorspace
            )
        elif meta.pdf_path is not None:  # Converted to PDF
            processing_fn = partial(
                process_pdf, pdf_path=meta.pdf_path, colorspace=colorspace
            )
        elif meta.extension in PIL_SUPPORTED_EXTENSIONS:
            processing_fn = partial(process_image, image_path=meta.path)
        else:
            logger.info("file type ending with %s not supported.", meta.extension)
            raise UnsupportedDocExtension(meta.extension, SUPPORTED_DOC_EXTS_LIST)
        pages = processing_fn(output_dir=output_dir)
    except REPORTED_ERRORS as e:
        logger.exception("error while processing %s", meta)
        report = ProcessingReport(
            doc_path=meta.path,
            pdf_path=meta.pdf_path,
            error=Error.from_exception(e),
        )
        return report
    report = ProcessingReport(doc_path=meta.path, pdf_path=meta.pdf_path, pages=pages)
    return report


@lru_cache(maxsize=1000)
def hash_filepath(filepath: Path) -> str:
    return hashlib.md5(str(filepath).encode()).hexdigest()


def make_page_filename(parent_doc: Path, page: int) -> str:
    parent_hash = hash_filepath(parent_doc)
    filename_without_ext, ext = os.path.splitext(parent_doc.name)
    if ext:
        ext = ext[1:]
    return f"{parent_hash}_{filename_without_ext}_{ext}_page_{page}{PNG_EXT}'"


def make_pdf_filename(source_path: Path) -> str:
    parent_hash = hash_filepath(source_path)
    filename_without_ext, ext = os.path.splitext(source_path.name)
    if ext:
        ext = ext[1:]
    return f"{parent_hash}_{filename_without_ext}_{ext}_converted{PDF_EXT}"


def process_pdf(
    pdf_path: Path, *, output_dir: Path, colorspace: Colorspace
) -> list[Path]:
    pdf_bytes = pdf_path.read_bytes()
    mode = COLOR_LUT[colorspace]
    pages = []
    try:
        with pymupdf.open("pdf", pdf_bytes) as pdf_doc:
            if pdf_doc.is_encrypted:
                raise InvalidPDF(pdf_path, reason="PDF is encrypted")
            for page_i in range(pdf_doc.page_count):
                page = pdf_doc.load_page(page_i)
                page_filename = make_page_filename(pdf_path, page_i)
                page_path = output_dir / page_filename
                if page_path.exists():
                    try:
                        Image.open(page_path, formats=(PIL_PNG,))
                        logger.debug("valid page image found at: %s", page_path)
                        pages.append(page_path)
                        continue
                    except UnidentifiedImageError:
                        pass
                pix = page.get_pixmap(colorspace=mode)
                # convert pixmap to pillow image
                # https://github.com/pymupdf/PyMuPDF/issues/322
                img = Image.frombytes(mode, (pix.w, pix.h), pix.samples)
                img.save(page_path, format=PIL_PNG)
                pages.append(page_path)
    except (EmptyFileError, FileDataError) as e:
        raise InvalidPDF(pdf_path) from e
    return pages


async def _convert_to_pdf(
    docs: Iterable[tuple[int, DocMetadata]],
    client: GotenbergClient,
    max_concurrency: int,
) -> AsyncGenerator[tuple[int, Path, bytes], None]:
    aws = (_pdf_conversion_wrapper(doc, doc_ix, client) for doc_ix, doc in docs)
    async for res in run_with_concurrency(aws, max_concurrency):
        yield res


async def _pdf_conversion_wrapper(
    doc: DocMetadata,
    doc_ix: int,
    client: GotenbergClient,
) -> tuple[int, Path, bytes]:
    doc_bytes = doc.path.read_bytes()
    converted = await client.convert_doc_to_pdf(doc_bytes, doc.extension)
    return doc_ix, doc.path, converted


def process_image(image_path: Path, *, output_dir: Path) -> list[Path]:
    pages = []
    try:
        with image_path.open("rb") as f:
            im = Image.open(f)
            has_multiple_images = (
                hasattr(im, "tag") and ("ImageDescription" in im.tag)
            ) or (hasattr(im, "is_animated") and im.is_animated)
            if has_multiple_images:
                for frame_i in range(im.n_frames):
                    # Iterate over each frame
                    filename = make_page_filename(image_path, frame_i)
                    page_path = output_dir / filename
                    im.seek(frame_i)
                    save_rgb_image(im, page_path)
                    pages.append(page_path)
            else:
                filename = make_page_filename(image_path, 0)
                page_path = output_dir / filename
                save_rgb_image(im, page_path)
                pages.append(page_path)
    except UnidentifiedImageError as e:
        raise InvalidImage(image_path) from e
    return pages


def save_rgb_image(im: Image, page_path: Path) -> None:
    if im.mode != "RGB":
        im = im.convert("RGB")
    im.save(page_path, PIL_PNG)
