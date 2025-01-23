# ruff: noqa: UP007
from __future__ import annotations

import json
import logging
import os
import traceback
import uuid
from abc import ABC
from pathlib import Path
from typing import Any, Callable, NoReturn, Optional, Union

from icij_common.pydantic_utils import ICIJModel
from icij_worker import TaskState
from icij_worker.objects import ErrorEvent, Registrable, Task, TaskError
from icij_worker.typing_ import AbstractSetIntStr, DictStrAny, MappingIntStrAny

from passport_service.constants import GOTENBERG_SUPPORTED_EXTS

try:
    from passport_service.typing_ import PassportEyeMRZ
except ImportError:
    PassportEyeMRZ = None

BoundingBox = tuple[float, float, float, float]


logger = logging.getLogger(__name__)


class DocMetadata(ICIJModel):
    path: Path
    extension: str
    pdf_path: Optional[Path] = None

    def relative_to(self, *, data_dir: Path) -> DocMetadata:
        path = data_dir / self.path
        pdf_path = self.pdf_path
        if pdf_path is not None:
            pdf_path = data_dir / pdf_path
        return DocMetadata(path=path, extension=self.extension, pdf_path=pdf_path)


def _id_title(title: str) -> str:
    id_title = []
    for i, letter in enumerate(title):
        if i and letter.isupper():
            id_title.append("-")
        id_title.append(letter.lower())
    return "".join(id_title)


class Error(ICIJModel):
    id: str
    title: str
    detail: str

    @classmethod
    def from_exception(cls, exception: BaseException) -> Error:
        title = exception.__class__.__name__
        trace_lines = traceback.format_exception(
            None, value=exception, tb=exception.__traceback__
        )
        detail = f"{exception}\n{''.join(trace_lines)}"
        error_id = f"{_id_title(title)}-{uuid.uuid4().hex}"
        error = Error(id=error_id, title=title, detail=detail)
        return error


class DetectionRequest(ICIJModel):
    # noqa: UP007
    doc_path: Path
    pdf_path: Optional[Path] = None
    pages: Optional[list[Path]] = None
    error: Optional[Error] = None

    def relative_to(self, *, data_dir: Path | None, work_dir: Path) -> DetectionRequest:
        doc_path = data_dir / self.doc_path
        pdf_path = self.pdf_path
        if pdf_path is not None:
            pdf_path = work_dir / pdf_path
        pages = self.pages
        if pages is not None:
            pages = [work_dir / p for p in pages]
        return DetectionRequest(
            doc_path=doc_path, pdf_path=pdf_path, pages=pages, error=self.error
        )


class ProcessingReport(DetectionRequest):
    def relative_to(self, *, data_dir: Path | None, work_dir: Path) -> ProcessingReport:
        doc_path = self.doc_path.relative_to(data_dir)
        pdf_path = self.pdf_path
        if pdf_path is not None:
            pdf_path = pdf_path.relative_to(work_dir)
        pages = self.pages
        if pages is not None:
            pages = [p.relative_to(work_dir) for p in pages]
        return ProcessingReport(doc_path=doc_path, pdf_path=pdf_path, pages=pages)


class ProcessingResponse(ICIJModel):
    detection_task_id: Optional[str]
    reports: list[ProcessingReport]


class ObjectDetection(ICIJModel):
    # noqa: UP007
    class_id: str
    confidence: float
    box: BoundingBox
    scale: Optional[float] = None


class MRZ(ICIJModel):
    country: Optional[str] = None
    metadata: dict[str, Any]

    # TODO: add raw_text, country, date_of_birth, sex, names, check_date_of_birth...
    # see PassportEyeMRZ.to_dict() for all available field

    @classmethod
    def from_passport_eye(
        cls, mrz: PassportEyeMRZ, mrz_to_country: Callable[[PassportEyeMRZ], str]
    ) -> MRZ:
        country = mrz_to_country(mrz)
        return cls(country=country, metadata=mrz.to_dict())


class Passport(ObjectDetection):
    mrz: Optional[MRZ] = None

    @classmethod
    def from_detection(cls, detection: ObjectDetection, mrz: MRZ = None) -> Passport:
        return cls(**detection.dict(), mrz=mrz)


class DocPageDetection(ICIJModel):
    page: int
    passports: list[Passport]


class PassportDetection(ICIJModel):
    doc_path: Path
    doc_pages: Optional[list[DocPageDetection]] = None
    error: Optional[Error] = None

    def relative_to(self, *, data_dir: Path) -> PassportDetection:
        doc_path = self.doc_path.relative_to(data_dir)
        return PassportDetection(doc_path=doc_path, doc_pages=self.doc_pages)


def generate_task_id(task_name: str) -> str:
    uid = uuid.uuid4()
    return f"{task_name}-{uid.hex}"


class TaskSearch(ICIJModel):
    name: Optional[str] = None
    status: Optional[Union[list[TaskState], TaskState]] = None


class PreprocessingRequest(ICIJModel):
    docs: list[DocMetadata]
    detection_args: dict[str, Any]


class PreprocessingTaskRequest(PreprocessingRequest):
    docs: list[DocMetadata] | Path
    batch_size: Optional[int] = 64
    detection_args: dict[str, Any]


def _raise(err: OSError) -> NoReturn:
    raise err


JSON_EXT = ".json"


def as_doc_metadata(
    docs_dir: Path,
    *,
    data_dir: Path,
    supported_ext: set[str] | None = None,
) -> list[DocMetadata]:
    if supported_ext is None:
        from passport_service.core.preprocessing import SUPPORTED_DOC_EXTS

        supported_ext = set(GOTENBERG_SUPPORTED_EXTS)
        supported_ext.update(SUPPORTED_DOC_EXTS)
    docs = []
    for root, _, files in os.walk(docs_dir, onerror=_raise):
        root = Path(root)  # noqa: PLW2901
        dir_docs = set()
        for f in files:
            ext = Path(f).suffix
            if ext and ext not in supported_ext:
                continue
            path = root.relative_to(data_dir) / f
            dir_docs.add((path, path.suffix))
        without_ext = [(p, ext) for p, ext in dir_docs if not ext]
        for doc in without_ext:
            dir_docs.remove(doc)
            path, _ = doc
            json_meta_path = data_dir / path.with_suffix(JSON_EXT)
            if json_meta_path.exists():
                ext = _read_extension_from_meta(json_meta_path)
                if ext is not None:
                    dir_docs.add((path, ext))
        dir_docs = (DocMetadata(path=p, extension=ext) for p, ext in dir_docs)
        docs.extend(dir_docs)
    docs = sorted(docs, key=lambda x: x.path)
    return docs


def _read_extension_from_meta(meta_path: Path) -> str | None:
    meta = json.loads(meta_path.read_text())
    meta = meta.get("metadata")
    if meta:
        resource_name = meta.get("tika_metadata_resourcename")
        if resource_name is not None:
            ext = os.path.splitext(resource_name)[1]
            if ext:
                return ext
    return None


class PassportDetectionRequest(ICIJModel):
    inputs: list[DetectionRequest]
    model_path: Path
    read_mrz: bool = True


# TODO: remove when switching to Pydantic v2. This class is copy of icij_worker.Task.
#  FastAPI use a hack for Pydantic v1 and v2 compat. The problem is that this hack
#  creates a brand new class from existing Pydantic models using the
#  `create_cloned_field` function. Since a new task class if created the base class
#  RegistrableMixing._registry is lost and all function which uses that registry fail.
#  In particular icij_worker.Task.dict fails and hence the response serialization


class _FastAPIRegistrableBugFix(Registrable, ICIJModel, ABC):
    def dict(
        self,
        *,
        include: AbstractSetIntStr | MappingIntStrAny | None = None,
        exclude: AbstractSetIntStr | MappingIntStrAny | None = None,
        by_alias: bool = False,
        skip_defaults: bool | None = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ) -> DictStrAny:
        return super(Registrable, self).dict(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            skip_defaults=skip_defaults,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )


class Task_(_FastAPIRegistrableBugFix, Task): ...  # noqa: N801


class TaskError_(_FastAPIRegistrableBugFix, TaskError): ...  # noqa: N801


class ErrorEvent_(_FastAPIRegistrableBugFix, ErrorEvent):  # noqa: N801
    error: TaskError_
