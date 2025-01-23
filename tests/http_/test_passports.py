from asyncio import Task
from copy import deepcopy
from pathlib import Path
from typing import NoReturn

import pytest
from icij_worker.exceptions import TaskQueueIsFull
from starlette.testclient import TestClient


@pytest.fixture
def paths_request() -> dict:
    payload = {
        "docs": [
            {"path": "passports/not_a_passport.jpg", "extension": ".jpg"},
            {"path": "passports/passport.docx", "extension": ".docx"},
            {"path": "passports/passport.odt", "extension": ".odt"},
            {"path": "passports/passport.pdf", "extension": ".pdf"},
            {"path": "passports/passport.png", "extension": ".png"},
        ],
        "detection_args": {"model": "some_model"},
    }
    return payload


@pytest.fixture
def dir_request(test_data_dir: Path) -> dict:
    payload = {
        "docs": str(test_data_dir),
        "detection_args": {"model": "some_model"},
    }
    return payload


@pytest.fixture(params=["paths_request", "dir_request"])
def test_preprocessing_payload(request) -> dict:  # noqa: ANN001
    return request.getfixturevalue(request.param)


@pytest.fixture
def test_create_preprocessing_task_payload(test_preprocessing_payload: dict) -> dict:
    payload = deepcopy(test_preprocessing_payload)
    payload["batch_size"] = 2
    return payload


def test_post_create_docs_preprocessing_tasks_should_return_201(
    test_client: TestClient, test_create_preprocessing_task_payload: dict
) -> None:
    # Given
    url = "/passports/preprocessing-tasks"
    payload = test_create_preprocessing_task_payload

    # When
    res = test_client.post(url, json=payload)

    # Then
    assert res.status_code == 201, res.json()
    assert res.text.startswith("create-preprocessing-tasks")


@pytest.mark.parametrize(
    "test_preprocessing_payload",
    ["paths_request"],
    indirect=["test_preprocessing_payload"],
)
def test_post_docs_preprocess00ing_should_return_201(
    test_client: TestClient, test_preprocessing_payload: dict
) -> None:
    # Given
    url = "/passports/preprocessing"
    payload = test_preprocessing_payload

    # When
    res = test_client.post(url, json=payload)

    # Then
    assert res.status_code == 201, res.json()
    assert res.text.startswith("preprocess-docs")


_ASYNC_APP_LIMITED_QUEUE = None


async def test_preprocessing_task_should_return_429_when_too_many_tasks(
    test_client: TestClient,
    tmp_path_factory: "TempPathFactory",  # noqa: F821
    monkeypatch,  # noqa: ANN001
) -> None:
    from passport_service.http_ import passports

    # Given
    url = "/passports/preprocessing"
    payload = {
        "docs": [
            {
                "path": str(tmp_path_factory.mktemp("passport-")),
                "extension": ".some_ext",
            }
        ],
        "detection_args": {"model": "some_model"},
    }

    class QueueIsFullTaskManager:
        async def save_task(self, task: Task) -> bool:  # noqa: ARG002
            return True

        async def enqueue(self, task: Task) -> NoReturn:  # noqa: ARG002
            raise TaskQueueIsFull(0)

    # When
    res_0 = test_client.post(url, json=payload)
    assert res_0.status_code == 201, res_0.json()

    monkeypatch.setattr(passports, "lifespan_task_manager", QueueIsFullTaskManager)
    res_1 = test_client.post(url, json=payload)
    # Then
    assert res_1.status_code == 429, res_1.json()


def test_post_passport_detection_should_return_200(
    test_client: TestClient,
) -> None:
    # Given
    url = "/passports/detection"
    payload = {
        "inputs": [
            {
                "doc_path": "/some/doc/path",
                "pdf_path": "/some/pdf/path",
                "pages": ["page/0/path", "page/1/path"],
            }
        ],
        "model_path": "some-model-path",
        "read_mrz": True,
    }

    # When
    res = test_client.post(url, json=payload)

    # Then
    assert res.status_code == 201, res.json()
    assert res.text.startswith("detect-passports")


async def test_create_passport_detection_task_should_return_429_when_too_many_tasks(
    test_client: TestClient,
    monkeypatch,  # noqa: ANN001
) -> None:
    from passport_service.http_ import passports

    url = "/passports/detection"
    payload = {
        "inputs": [
            {
                "doc_path": "/some/doc/path",
                "pdf_path": "/some/pdf/path",
                "pages": ["page/0/path", "page/1/path"],
            }
        ],
        "model_path": "some-model-path",
        "read_mrz": True,
    }

    class QueueIsFullTaskManager:
        async def save_task(self, task: Task) -> bool:  # noqa: ARG002
            return True

        async def enqueue(self, task: Task) -> NoReturn:  # noqa: ARG002
            raise TaskQueueIsFull(0)

    # When
    res_0 = test_client.post(url, json=payload)
    assert res_0.status_code == 201, res_0.json()

    monkeypatch.setattr(passports, "lifespan_task_manager", QueueIsFullTaskManager)
    res_1 = test_client.post(url, json=payload)
    # Then
    assert res_1.status_code == 429, res_1.json()
