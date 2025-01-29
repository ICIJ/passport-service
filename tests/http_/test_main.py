from unittest.mock import AsyncMock

from _pytest.monkeypatch import MonkeyPatch
from starlette.testclient import TestClient


def test_health_should_return_200(
    test_client: TestClient,
) -> None:
    # Given
    url = "/health"
    # When
    res = test_client.get(url)
    # Then
    assert res.status_code == 200, res.json()
    health = res.json()
    assert health == {"amqp": True, "storage": True}


def test_health_should_return_503(
    test_client: TestClient, monkeypatch: MonkeyPatch
) -> None:
    # Given
    url = "/health"

    def _mock_task_manager() -> AsyncMock:
        mocked_tm = AsyncMock()
        mocked_tm.get_health.return_value = {"amqp": False, "storage": True}
        return mocked_tm

    monkeypatch.setattr(
        "passport_service.http_.main.lifespan_task_manager", _mock_task_manager
    )

    # When
    res = test_client.get(url)
    # Then
    assert res.status_code == 503, res.json()
    health = res.json()
    assert health == {"amqp": False, "storage": True}
