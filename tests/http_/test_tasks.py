# ruff: noqa: DTZ005
from datetime import datetime

from icij_worker import Task, TaskState
from icij_worker.objects import ErrorEvent, ResultEvent, StacktraceItem, TaskError
from starlette.testclient import TestClient

from passport_service.http_ import tasks


async def test_put_task(test_client: TestClient, monkeypatch) -> None:  # noqa: ANN001
    # Given
    task_id = "some_id"
    task = {"name": "some-task", "args": []}

    class _MockManager:
        async def save_task(self, task: Task) -> bool:  # noqa: ARG002
            is_new = True
            return is_new

        async def enqueue(self, task: Task):  # noqa: ANN202
            pass

    monkeypatch.setattr(tasks, "lifespan_task_manager", _MockManager)
    task_url = f"/tasks/{task_id}"

    # When
    res = test_client.put(task_url, json=task)
    assert res.status_code == 201, res.json()
    assert res.text == task_id


async def test_get_task(test_client: TestClient, monkeypatch) -> None:  # noqa: ANN001
    # Given
    task = Task(
        id="some_task",
        name="some_task_name",
        state=TaskState.CREATED,
        created_at=datetime.now(),
        args=dict(),
        max_retries=3,
        retries_left=3,
    )

    class _MockManager:
        async def get_task(self, task_id: str) -> Task:  # noqa: ARG002
            return task

    monkeypatch.setattr(tasks, "lifespan_task_manager", _MockManager)
    task_url = "/tasks/some-id"

    # When
    res = test_client.get(task_url)
    assert res.status_code == 200, res.json()
    expected_task = {
        "args": dict(),
        "completedAt": None,
        "createdAt": task.created_at.isoformat(),
        "id": "some_task",
        "name": "some_task_name",
        "progress": None,
        "retriesLeft": 3,
        "maxRetries": 3,
        "state": "CREATED",
    }
    assert res.json() == expected_task


def test_get_task_should_return_404_for_unknown_task(test_client: TestClient) -> None:
    # Given
    url = "/tasks/idontexist"
    # When
    res = test_client.get(url)
    # Then
    assert res.status_code == 404, res.json()
    error = res.json()
    assert error["detail"] == 'Unknown task "idontexist"'


def test_get_task_error(test_client: TestClient, monkeypatch) -> None:  # noqa: ANN001
    # Given
    error = ErrorEvent(
        task_id="some_task_id",
        error=TaskError(
            name="some error",
            message="some message",
            cause=None,
            stacktrace=[
                StacktraceItem(name="some error", file="some_file", lineno=666)
            ],
        ),
        created_at=datetime.now(),
        retries_left=3,
    )

    class _MockManager:
        async def get_task_errors(self, task_id: str) -> list[ErrorEvent]:  # noqa: ARG002
            return [error]

    url = f"/tasks/{error.task_id}/errors"
    monkeypatch.setattr(tasks, "lifespan_task_manager", _MockManager)

    # When
    res = test_client.get(url)
    assert res.status_code == 200, res.json()

    # Then
    errors = res.json()
    assert len(errors) == 1
    assert errors[0]["error"]["name"] == "some error"


def test_get_task_error_should_return_404_for_unknown_task(
    test_client: TestClient,
) -> None:
    # Given
    url = "/tasks/idontexist/error"
    # When
    res = test_client.get(url)
    # Then
    assert res.status_code == 404, res.json()
    error = res.json()
    assert error["detail"] == "Not Found"


def test_get_task_result(test_client: TestClient, monkeypatch) -> None:  # noqa: ANN001
    # Given
    result = "some_result"
    result_event = ResultEvent(
        task_id="some_task_id", created_at=datetime.now(), result=result
    )
    url = f"/tasks/{result_event.task_id}/result"

    class _MockManager:
        async def get_task_result(self, task_id: str) -> ResultEvent:  # noqa: ARG002
            return result_event

    monkeypatch.setattr(tasks, "lifespan_task_manager", _MockManager)

    # When
    res = test_client.get(url)
    assert res.status_code == 200, res.json()

    # Then
    res = res.json()
    assert res == result


def test_get_task_result_should_return_404_for_unknown_task(
    test_client: TestClient,
) -> None:
    # Given
    url = "/tasks/idontexist/result"
    # When
    res = test_client.get(url)
    # Then
    assert res.status_code == 404, res.json()
    error = res.json()
    assert error["detail"] == 'Unknown task "idontexist"'
