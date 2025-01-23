# ruff: noqa: ANN001
import asyncio
import os
import shutil
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path

import aiohttp
import pytest
from aiohttp import ClientTimeout

# noinspection PyUnresolvedReferences
from icij_common.test_utils import reset_env  # noqa: F401
from icij_worker import AMQPWorkerConfig, TaskState
from starlette.testclient import TestClient

from passport_service.app import app
from passport_service.config import AppConfig
from passport_service.core.pdf_conversion import GotenbergClient
from tests import TEST_DATA_DIR

RABBITMQ_TEST_PORT = 5672
RABBITMQ_TEST_HOST = "localhost"
RABBITMQ_DEFAULT_VHOST = "%2F"

_RABBITMQ_MANAGEMENT_PORT = 15672
TEST_MANAGEMENT_URL = f"http://localhost:{_RABBITMQ_MANAGEMENT_PORT}"
_DEFAULT_BROKER_URL = (
    f"amqp://guest:guest@{RABBITMQ_TEST_HOST}:{RABBITMQ_TEST_PORT}/"
    f"{RABBITMQ_DEFAULT_VHOST}"
)
_DEFAULT_AUTH = aiohttp.BasicAuth(login="guest", password="guest", encoding="utf-8")


@pytest.fixture(scope="session")
def event_loop(request) -> Iterator[asyncio.AbstractEventLoop]:  # noqa: ARG001
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_model_v0() -> Path:
    model_filename = "test_model_v0.onnx"
    test_model_path = TEST_DATA_DIR / "models" / model_filename
    if not test_model_path.exists():
        raise ValueError("place a model in here or implement automatic model DL")
    return test_model_path


@pytest.fixture(scope="session")
def test_work_dir_session(tmpdir_factory) -> Path:
    return Path(tmpdir_factory.mktemp("passport_workdir"))


@pytest.fixture
def test_work_dir(test_work_dir_session: Path) -> Path:
    for path in test_work_dir_session.iterdir():
        if path.is_file():
            os.unlink(path)
        else:
            shutil.rmtree(path)
    return test_work_dir_session


@pytest.fixture(scope="session")
def http_service_address() -> str:
    return "http://testserver"


@pytest.fixture(scope="session")
def test_app_config(
    test_work_dir_session: Path, http_service_address: str
) -> AppConfig:
    country_codes = ["EOL", "FRA"]
    return AppConfig(
        n_mp_workers=2,
        preprocessing_batch_size=1,
        data_dir=test_work_dir_session / "data",
        work_dir=test_work_dir_session,
        country_codes=country_codes,
        http_service_address=http_service_address,
        test=True,
    )


@pytest.fixture(scope="session")
async def test_gotenberg_client(test_app_config: AppConfig) -> GotenbergClient:
    client = test_app_config.to_gotenberg_client()
    async with client:
        yield client


@pytest.fixture(scope="session")
def test_app_config_path(tmpdir_factory, test_app_config: AppConfig) -> Path:
    config_path = Path(tmpdir_factory.mktemp("app_config")).joinpath("app_config.json")
    config_path.write_text(test_app_config.json())
    return config_path


@pytest.fixture(scope="session")
def test_worker_config(test_app_config_path: Path) -> AMQPWorkerConfig:
    return AMQPWorkerConfig(
        log_level="DEBUG", app_bootstrap_config_path=test_app_config_path
    )


@pytest.fixture(scope="session")
async def with_worker_lifespan_deps(
    test_worker_config: AMQPWorkerConfig,
) -> AsyncGenerator[None, None]:
    worker_id = "test-worker-id"
    async with app.lifetime_dependencies(
        worker_config=test_worker_config, worker_id=worker_id
    ):
        yield


@pytest.fixture(scope="session")
async def rabbit_mq_session() -> str:
    await wipe_rabbit_mq()
    return _DEFAULT_BROKER_URL


@pytest.fixture
async def rabbit_mq() -> str:
    await wipe_rabbit_mq()
    return _DEFAULT_BROKER_URL


def rabbit_mq_test_session() -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        raise_for_status=True, auth=_DEFAULT_AUTH, timeout=ClientTimeout(total=2)
    )


@pytest.fixture
def test_data_dir(test_work_dir: Path) -> Path:
    dir_name = TEST_DATA_DIR.name
    shutil.copytree(TEST_DATA_DIR, test_work_dir / dir_name)
    return test_work_dir / dir_name


async def wipe_rabbit_mq() -> None:
    async with rabbit_mq_test_session() as session:
        await _delete_all_connections(session)
        tasks = [_delete_all_exchanges(session), _delete_all_queues(session)]
        await asyncio.gather(*tasks)


def get_test_management_url(url: str) -> str:
    return f"{TEST_MANAGEMENT_URL}{url}"


async def _delete_all_connections(session: aiohttp.ClientSession) -> None:
    async with session.get(get_test_management_url("/api/connections")) as res:
        connections = await res.json()
        tasks = [_delete_connection(session, conn["name"]) for conn in connections]
    await asyncio.gather(*tasks)


async def _delete_connection(session: aiohttp.ClientSession, name: str) -> None:
    async with session.delete(get_test_management_url(f"/api/connections/{name}")):
        pass


async def _delete_all_exchanges(session: aiohttp.ClientSession) -> None:
    url = f"/api/exchanges/{RABBITMQ_DEFAULT_VHOST}"
    async with session.get(get_test_management_url(url)) as res:
        exchanges = list(await res.json())
        exchanges = (
            ex for ex in exchanges if ex["user_who_performed_action"] == "guest"
        )
        tasks = [_delete_exchange(session, ex["name"]) for ex in exchanges]
    await asyncio.gather(*tasks)


async def _delete_exchange(session: aiohttp.ClientSession, name: str) -> None:
    url = f"/api/exchanges/{RABBITMQ_DEFAULT_VHOST}/{name}"
    async with session.delete(get_test_management_url(url)):
        pass


async def _delete_all_queues(session: aiohttp.ClientSession) -> None:
    url = f"/api/queues/{RABBITMQ_DEFAULT_VHOST}"
    async with session.get(get_test_management_url(url)) as res:
        queues = await res.json()
    tasks = [_delete_queue(session, q["name"]) for q in queues]
    await asyncio.gather(*tasks)


async def _delete_queue(session: aiohttp.ClientSession, name: str) -> None:
    url = f"/api/queues/{RABBITMQ_DEFAULT_VHOST}/{name}"
    async with session.delete(get_test_management_url(url)) as res:
        res.raise_for_status()


def assert_task_has_state(
    client: TestClient, task_id: str, expected_state: TaskState
) -> bool:
    url = f"/tasks/{task_id}/state"
    res = client.get(url)
    if res.status_code != 200:
        raise ValueError(res.json())
    state = TaskState(res.text)
    if state is TaskState.ERROR and expected_state is not TaskState.ERROR:
        raise AssertionError(f"Task failed while expecting {expected_state}")
    return state is expected_state


def all_done(client: TestClient, task_ids: list[str]) -> bool:
    return all(
        assert_task_has_state(client, task_id, TaskState.DONE) for task_id in task_ids
    )
