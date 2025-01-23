from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiohttp.typedefs import StrOrURL
from icij_common.pydantic_utils import jsonable_encoder
from pydantic import parse_obj_as

from passport_service.app import (
    create_preprocessing_tasks,
    detect_passports,
    preprocess_docs,
)
from passport_service.objects import (
    MRZ,
    DocMetadata,
    PassportDetection,
    as_doc_metadata,
)


@pytest.mark.integration
async def test_passport_pipeline(
    with_worker_lifespan_deps,  # noqa: ANN001, ARG001
    test_model_v0: Path,
    test_data_dir: Path,
    monkeypatch,  # noqa: ANN001, ARG001
) -> None:
    # Given
    batch_size = 64
    detection_args = {"model_path": test_model_v0, "read_mrz": True}
    data_dir = test_data_dir
    docs_dir = test_data_dir / "passports"

    @asynccontextmanager
    async def _put_and_assert_preprocessing_task_creation(
        _,  # noqa: ANN001
        url: StrOrURL,
        *,
        data: Any = None,
        **kwargs: Any,  # noqa: ANN001
    ) -> AbstractAsyncContextManager[None]:
        assert url.startswith("/tasks/preprocess-docs-")
        expected_docs = [
            DocMetadata(path=Path("passports", "not_a_passport"), extension=".jpg"),
            DocMetadata(path=Path("passports", "not_a_passport.jpg"), extension=".jpg"),
            DocMetadata(path=Path("passports", "passport.docx"), extension=".docx"),
            DocMetadata(path=Path("passports", "passport.odt"), extension=".odt"),
            DocMetadata(path=Path("passports", "passport.pdf"), extension=".pdf"),
            DocMetadata(path=Path("passports", "passport.png"), extension=".png"),
        ]
        expected_task = {
            "name": "preprocess-docs",
            "args": {"docs": expected_docs, "detection_args": detection_args},
        }
        expected_task = jsonable_encoder(expected_task)
        expected_data = expected_task
        assert data is None
        json_data = kwargs.pop("json")
        assert not kwargs
        assert json_data == expected_data
        mocked_res = AsyncMock()
        mocked_res.text.return_value = "create-preprocessing-tasks-some-id"
        yield mocked_res

    monkeypatch.setattr(
        "passport_service.http_.task_client.TaskClient._put",
        _put_and_assert_preprocessing_task_creation,
    )
    # When
    create_tasks_task_ids = await create_preprocessing_tasks(
        str(docs_dir), batch_size=batch_size, detection_args=detection_args
    )
    # Then
    assert create_tasks_task_ids == ["create-preprocessing-tasks-some-id"]

    # Given
    request = as_doc_metadata(data_dir=data_dir, docs_dir=docs_dir)
    request = [r.dict(by_alias=True) for r in request]

    # When
    @asynccontextmanager
    async def _put_and_assert_detection_tasks_creation(
        _,  # noqa: ANN001
        url: StrOrURL,
        *,
        data: Any = None,
        **kwargs: Any,  # noqa: ANN001
    ) -> AbstractAsyncContextManager[None]:
        assert url.startswith("/tasks/detect-passports-")
        assert data is None
        json_data = kwargs.pop("json")
        assert not kwargs
        assert json_data["name"] == "detect-passports"
        assert json_data["args"]["model_path"] == str(test_model_v0)
        assert json_data["args"]["read_mrz"] is True
        assert len(json_data["args"]["inputs"]) == 6
        mocked_res = AsyncMock()
        mocked_res.text.return_value = "detect-passports-some-id"
        yield mocked_res

    monkeypatch.setattr(
        "passport_service.http_.task_client.TaskClient._put",
        _put_and_assert_detection_tasks_creation,
    )

    preprocess_response = await preprocess_docs(
        docs=request, detection_args=detection_args
    )
    reports = preprocess_response["reports"]
    assert len(reports) == 6
    assert not any(r.get("error") for r in reports)
    assert preprocess_response["detection_task_id"] == "detect-passports-some-id"

    # When
    detections = await detect_passports(reports, **detection_args)
    detections = parse_obj_as(list[PassportDetection], detections)
    detections = sorted(detections, key=lambda x: x.doc_path)
    # Then
    assert len(detections) == 6
    not_passport_0 = detections[0]
    assert not_passport_0.doc_path.name == "not_a_passport"
    assert not not_passport_0.doc_pages
    not_passport_1 = detections[1]
    assert not_passport_1.doc_path.name == "not_a_passport.jpg"
    assert not not_passport_1.doc_pages

    passport_png = detections[5]
    assert passport_png.doc_path.name == "passport.png"
    assert len(passport_png.doc_pages) == 1
    first_page = passport_png.doc_pages[0]
    assert first_page.page == 0
    doc_page = first_page.passports
    assert len(doc_page) == 2
    first_passport = doc_page[0]
    assert first_passport.confidence == pytest.approx(0.903170, abs=1e-5)
    first_box = (2.618, 0.026, 541.175, 257.426)
    assert first_passport.box == pytest.approx(first_box, abs=1e-3)
    assert first_passport.mrz is None
    second_passport = doc_page[1]
    assert second_passport.confidence == pytest.approx(0.873809, abs=1e-5)
    second_box = (4.076, 240.260, 541.396, 395.676)
    assert second_passport.box == pytest.approx(second_box, abs=1e-3)
    mrz = second_passport.mrz
    assert isinstance(mrz, MRZ)
    assert mrz.country == "EOL"
    assert mrz.metadata["names"] == "JANE"
    assert mrz.metadata["surname"] == "SMITH"

    for i, path in zip([2, 3, 4], ["passport.docx", "passport.odt", "passport.pdf"]):
        passport = detections[i]
        assert passport.doc_path.name == path
        assert len(passport.doc_pages) == 1
        doc_page = passport.doc_pages[0]
        assert doc_page.page == 0
        assert len(doc_page.passports) == 2
        assert any(p.mrz for p in doc_page.passports)
