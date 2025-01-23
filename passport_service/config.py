import csv
from concurrent.futures import ProcessPoolExecutor
from functools import cached_property
from multiprocessing import Pool
from pathlib import Path
from typing import ClassVar, Generic, Optional

import icij_worker
from icij_common.pydantic_utils import ICIJSettings, NoEnumModel
from icij_worker.utils.config import TM, SettingsWithTM
from icij_worker.utils.logging_ import LogWithWorkerIDMixin
from pydantic import Field

try:
    from passport_service.core.pdf_conversion import GotenbergClient
except ImportError:
    GotenbergClient = None

try:
    from passport_service.http_ import TaskClient
except ImportError:
    TaskClient = None

import passport_service

_ALL_LOGGERS = [passport_service.__name__, icij_worker.__name__]


class AppConfig(ICIJSettings, LogWithWorkerIDMixin, NoEnumModel):
    class Config:
        env_prefix = "PASSPORT_ASYNC_"

    loggers: ClassVar[list[str]] = Field(_ALL_LOGGERS, const=True)

    country_codes: Optional[list[str]] = None
    data_dir: Path

    gotenberg_url: str = "http://localhost:3000"
    gotenberg_max_retries: int = 5
    gotenberg_min_retry_wait_s: float = 5.0
    gotenberg_max_retry_wait_s: float = 30.0
    gotenberg_max_retry_randomness_s: float = 2.0

    http_service_address: str = "http://localhost:8080"

    inference_batch_size: int = 32
    log_level: str = Field(default="INFO")
    n_mp_workers: Optional[int] = None
    passport_label: str = "passport"
    pdf_conversion_concurrency: int = 2
    preprocessing_batch_size: int = 10
    test: bool = False
    work_dir: Path

    def to_process_pool_executor(self) -> Pool:
        # TODO: set maxtasksperchild to a fixed value to avoid blocking on exit ???
        return ProcessPoolExecutor(max_workers=self.n_mp_workers)

    @cached_property
    def countries(self) -> list[str]:
        if self.country_codes is not None:
            return self.country_codes
        csv_path = passport_service.DATA_DIR / "default_country_codes.csv"
        with csv_path.open() as csvfile:
            reader = csv.reader(csvfile)
            countries = [row[2] for row in reader]
        return countries

    def to_gotenberg_client(self) -> GotenbergClient:
        client = GotenbergClient(
            self.gotenberg_url,
            max_retries=self.gotenberg_max_retries,
            min_retry_wait_s=self.gotenberg_min_retry_wait_s,
            max_retry_wait_s=self.gotenberg_max_retry_wait_s,
            max_retry_randomness_s=self.gotenberg_max_retry_randomness_s,
        )
        return client

    def to_task_client(self) -> TaskClient:
        client = TaskClient(self.http_service_address)
        return client


class HttpServiceConfig(SettingsWithTM[TM], LogWithWorkerIDMixin, Generic[TM]):
    app_path: ClassVar[str] = Field(const=True, default="passport_service.app.app")
    loggers: ClassVar[list[str]] = Field(
        _ALL_LOGGERS + ["uvicorn", "__main__"], const=True
    )

    gunicorn_workers: int = 1
    log_level: str = Field(default="INFO")
    port: int = 8080

    class Config:
        env_prefix = "PASSPORT_HTTP_"
