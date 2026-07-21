import logging
from multiprocessing.pool import Pool

from icij_worker import WorkerConfig
from icij_worker.utils.dependencies import DependencyInjectionError

from ..config import AppConfig

try:
    from ..core.pdf_conversion import GotenbergClient
except ImportError:
    GotenbergClient = None

try:
    from ..http_ import TaskClient
except ImportError:
    TaskClient = None

logger = logging.getLogger(__name__)

_ASYNC_APP_CONFIG: AppConfig | None = None
_PROCESS_POOL_EXECUTOR: Pool | None = None
_GOTENBERG_CLIENT: GotenbergClient | None = None
_TASK_CLIENT: TaskClient | None = None


def load_app_config(worker_config: WorkerConfig, **_) -> None:
    global _ASYNC_APP_CONFIG
    if worker_config.app_bootstrap_config_path is not None:
        _ASYNC_APP_CONFIG = AppConfig.model_validate_json(
            worker_config.app_bootstrap_config_path.read_text()
        )
    else:
        _ASYNC_APP_CONFIG = AppConfig()


def setup_loggers(worker_id: str, **_) -> None:
    config = lifespan_config()
    config.setup_loggers(worker_id=worker_id)
    logger.info("worker loggers ready to log 💬")
    logger.info("app config: %s", config.model_dump_json(indent=2))


def process_pool_setup(**_) -> None:
    global _PROCESS_POOL_EXECUTOR
    config = lifespan_config()
    executor = config.to_process_pool_executor()
    executor.__enter__()
    _PROCESS_POOL_EXECUTOR = executor


def process_pool_teardown(exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
    global _PROCESS_POOL_EXECUTOR
    _PROCESS_POOL_EXECUTOR.__exit__(exc_type, exc_val, exc_tb)
    _PROCESS_POOL_EXECUTOR = None


async def gotenberg_client_setup(**_) -> None:
    global _GOTENBERG_CLIENT
    config = lifespan_config()
    client = config.to_gotenberg_client()
    await client.__aenter__()
    _GOTENBERG_CLIENT = client


async def gotenberg_client_teardown(exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
    global _GOTENBERG_CLIENT
    await _GOTENBERG_CLIENT.__aexit__(exc_type, exc_val, exc_tb)
    _GOTENBERG_CLIENT = None


async def task_client_setup(**_) -> None:
    global _TASK_CLIENT
    config = lifespan_config()
    client = config.to_task_client()
    await client.__aenter__()
    _TASK_CLIENT = client


async def task_client_teardown(exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
    global _TASK_CLIENT
    await _TASK_CLIENT.__aexit__(exc_type, exc_val, exc_tb)
    _TASK_CLIENT = None


def lifespan_config() -> AppConfig:
    if _ASYNC_APP_CONFIG is None:
        raise DependencyInjectionError("config")
    return _ASYNC_APP_CONFIG


def lifespan_pool_executor() -> Pool:
    if _PROCESS_POOL_EXECUTOR is None:
        raise DependencyInjectionError("multiprocessing pool")
    return _PROCESS_POOL_EXECUTOR


def lifespan_gotenberg_client() -> GotenbergClient:
    if _GOTENBERG_CLIENT is None:
        raise DependencyInjectionError("Gotenberg client")
    return _GOTENBERG_CLIENT


def lifespan_task_client() -> TaskClient:
    if _TASK_CLIENT is None:
        raise DependencyInjectionError("task client")
    return _TASK_CLIENT


APP_LIFESPAN_DEPS = [
    ("loading async app configuration", load_app_config, None),
    ("loggers", setup_loggers, None),
    ("process pool executor", process_pool_setup, process_pool_teardown),
    ("Gotenberg client", gotenberg_client_setup, gotenberg_client_teardown),
    ("task client", task_client_setup, task_client_teardown),
]
