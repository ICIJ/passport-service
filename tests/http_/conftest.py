# ruff: noqa: ANN001
import logging
from collections.abc import AsyncGenerator, Generator
from multiprocessing import Process

import pytest
from fastapi import FastAPI
from icij_worker import AMQPWorkerConfig, PostgresStorageConfig, WorkerBackend
from icij_worker.task_manager.amqp import AMQPTaskManagerConfig
from psycopg import AsyncConnection
from starlette.testclient import TestClient

from passport_service.config import HttpServiceConfig
from passport_service.http_.dependencies import lifespan_http_service_config
from passport_service.http_.service import create_service
from tests.conftest import (
    RABBITMQ_DEFAULT_VHOST,
    RABBITMQ_TEST_HOST,
    RABBITMQ_TEST_PORT,
    wipe_rabbit_mq,
)

logger = logging.getLogger(__name__)

POSTGRES_TEST_PORT = 5435


@pytest.fixture(scope="session")
def test_config(rabbit_mq_session: str) -> HttpServiceConfig:  # noqa: ARG001
    task_manager = AMQPTaskManagerConfig(
        rabbitmq_host=RABBITMQ_TEST_HOST,
        rabbitmq_port=RABBITMQ_TEST_PORT,
        rabbitmq_vhost=RABBITMQ_DEFAULT_VHOST,
        app_path="passport_service.app.app",
        storage=PostgresStorageConfig(port=POSTGRES_TEST_PORT),
    )
    config = HttpServiceConfig[AMQPTaskManagerConfig](task_manager=task_manager)
    return config


@pytest.fixture(scope="session")
def test_service_session(test_config: HttpServiceConfig) -> FastAPI:
    return create_service(test_config)


@pytest.fixture(scope="session")
def test_client_session(
    test_service_session: FastAPI, http_service_address: str
) -> TestClient:
    return TestClient(test_service_session, base_url=http_service_address)


async def _make_worker_process(
    worker_config: AMQPWorkerConfig, group: str | None
) -> Process:
    app = "passport_service.app.app"
    kwargs = {
        "n_workers": 1,
        "group": group,
        "app": app,
        "config": worker_config,
    }
    return Process(target=WorkerBackend.MULTIPROCESSING.run, kwargs=kwargs)


@pytest.fixture
async def test_preprocessing_worker(
    test_client: TestClient,  # noqa: ARG001
    test_worker_config: AMQPWorkerConfig,  # noqa: ARG001
) -> Generator[None, None, None]:
    # We depend on the test client to make sure the TM is created and sets up the
    # exchanges and queues
    p = await _make_worker_process(test_worker_config, "preprocessing")
    p.start()
    yield
    p.kill()  # Can't we do better, sigterm should be handled


@pytest.fixture
async def test_inference_worker(
    test_client: TestClient,  # noqa: ARG001
    test_worker_config: AMQPWorkerConfig,
) -> AsyncGenerator[None, None]:
    # We depend on the test client to make sure the TM is created and sets up the
    # exchanges and queues
    p = await _make_worker_process(test_worker_config, "inference")
    p.start()
    yield
    p.kill()  # Can't we do better, sigterm should be handled


async def _wipe_tasks(conn: AsyncConnection) -> None:
    async with conn.cursor() as cur:
        delete_everything = """TRUNCATE TABLE tasks CASCADE;
TRUNCATE TABLE results;
TRUNCATE TABLE errors;
"""
        try:  # noqa: SIM105
            await cur.execute(delete_everything)
        except:  # noqa: E722
            pass


@pytest.fixture
async def test_client(test_client_session: TestClient) -> TestClient:
    await wipe_rabbit_mq()
    with test_client_session:
        config = lifespan_http_service_config()
        conn_info = config.task_manager.storage.as_connection_info
        conn = await AsyncConnection.connect(autocommit=True, **conn_info.kwargs)
        async with conn:
            await _wipe_tasks(conn)
        yield test_client_session
