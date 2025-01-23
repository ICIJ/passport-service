from __future__ import annotations

import logging
from asyncio import get_running_loop
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from io import BytesIO
from typing import Any

from aiohttp import (
    ClientResponse,
    ClientResponseError,
    ClientSession,
    FormData,
    ServerDisconnectedError,
    ServerTimeoutError,
)
from aiohttp.client_exceptions import ClientConnectionResetError
from PIL.Image import EXTENSION
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    after_log,
    stop_after_attempt,
    wait_exponential,
)
from tenacity.wait import wait_base, wait_random

from passport_service.constants import (
    GOTENBERG_LIBREOFFICE_EXTS,
    GOTENBERG_SUPPORTED_EXTS,
    HTML_EXT,
    MARKDOWN_EXT,
)
from passport_service.exceptions import UnsupportedDocExtension

logger = logging.getLogger(__name__)

_RETRIED_STATUSES = {
    429,  # Too Many Requests
    503,  # Service Unavailable
}
_RETRIED_HTTP_EXCEPTIONS = (
    ClientConnectionResetError,
    ServerTimeoutError,
    ServerDisconnectedError,
)


class GotenbergClient:
    def __init__(
        self,
        service_url: str,
        max_retries: int = 5,
        min_retry_wait_s: float = 5.0,
        max_retry_wait_s: float = 30.0,
        max_retry_randomness_s: float = 2.0,
    ):
        self._service_url = service_url
        self._exit_stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._min_retry_wait_s = min_retry_wait_s
        self._max_retry_wait_s = max_retry_wait_s
        self._max_retry_randomness_s = max_retry_randomness_s
        self.convert_doc_to_pdf = AsyncRetrying(
            retry=self._retry_when,
            stop=stop_after_attempt(max_retries),
            wait=self._wait,
            reraise=True,
            after=after_log(logger, logging.ERROR),
        ).wraps(self.convert_doc_to_pdf)

    async def __aenter__(self) -> GotenbergClient:
        self._session = ClientSession(loop=get_running_loop())
        await self._exit_stack.enter_async_context(self._session)
        await self._session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    async def convert_doc_to_pdf(self, doc: bytes, ext: str) -> bytes:
        if ext == HTML_EXT:
            route = "/forms/chromium/convert/html"
        elif ext == MARKDOWN_EXT:
            route = "/forms/chromium/convert/markdown"
        elif ext in GOTENBERG_LIBREOFFICE_EXTS:
            route = "/forms/libreoffice/convert"
        else:
            raise UnsupportedDocExtension(ext, sorted(GOTENBERG_SUPPORTED_EXTS))
        data = FormData()
        filename = f"doc{ext}"
        data.add_field("form", BytesIO(doc), filename=filename)
        async with self._post(route, data=data) as response:
            converted = await response.read()
            return converted

    @asynccontextmanager
    async def _post(
        self, route: str, *, data: Any, **kwargs
    ) -> AsyncGenerator[ClientResponse, None]:
        url = self._url(route)
        async with self._session.post(url, data=data, **kwargs) as response:
            yield response

    def _url(self, route: str) -> str:
        return self._service_url + route

    @property
    def _wait(self) -> wait_base:
        wait = wait_exponential(
            multiplier=self._min_retry_wait_s,
            min=self._min_retry_wait_s,
            max=self._max_retry_wait_s,
        )
        wait += wait_random(min=0, max=self._max_retry_randomness_s)
        return wait

    @staticmethod
    def _retry_when(retry_state: RetryCallState) -> bool:
        exc = retry_state.outcome.exception()
        if isinstance(exc, ClientResponseError) and exc.status in _RETRIED_STATUSES:
            return True
        return isinstance(exc, _RETRIED_HTTP_EXCEPTIONS)


def should_convert_to_pdf(ext: str) -> bool:
    if ext in EXTENSION:
        return False
    convert = ext in GOTENBERG_SUPPORTED_EXTS
    return convert
