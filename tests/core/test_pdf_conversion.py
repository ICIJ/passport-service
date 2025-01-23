import pytest
from aiohttp import (
    ClientConnectionError,
    ClientConnectionResetError,
    ClientOSError,
    ClientResponseError,
    RequestInfo,
    ServerDisconnectedError,
    ServerTimeoutError,
)
from icij_common.test_utils import fail_if_exception
from multidict import CIMultiDict, CIMultiDictProxy
from pymupdf import pymupdf
from yarl import URL

from passport_service.config import AppConfig
from passport_service.core.pdf_conversion import GotenbergClient
from tests import TEST_DATA_DIR


async def test_convert_doc_to_pdf(test_gotenberg_client: GotenbergClient) -> None:
    # Given
    client = test_gotenberg_client
    passport_docx_path = TEST_DATA_DIR.joinpath("passports", "passport.docx")
    passport_docx = passport_docx_path.read_bytes()
    # When
    converted = await client.convert_doc_to_pdf(passport_docx, ext=".docx")
    # Then
    msg = "Converted PDF is invalid"
    with fail_if_exception(msg), pymupdf.open("pdf", converted) as pdf_doc:
        n_pages = sum(1 for _ in pdf_doc.pages())
    assert n_pages == 1


class MockFailingClient(GotenbergClient):
    def __init__(self, service_url: str, raised: Exception):
        super().__init__(
            service_url,
            max_retries=2,
            min_retry_wait_s=0.01,
            max_retry_wait_s=0.01,
            max_retry_randomness_s=0.001,
        )
        self._attempts = 0
        self._raised = raised

    async def convert_doc_to_pdf(self, doc: bytes, ext: str) -> bytes:  # noqa: ARG002
        self._attempts += 1
        if self._attempts == 1:
            raise self._raised
        return b"success"

    @property
    def attempts(self) -> int:
        return self._attempts


def _make_client_response_error(status: int) -> ClientResponseError:
    return ClientResponseError(
        RequestInfo(URL(""), "", CIMultiDictProxy(CIMultiDict()), URL("")),
        tuple(),
        status=status,
    )


@pytest.mark.parametrize(
    "raised",
    [
        ClientConnectionResetError(),
        ServerTimeoutError(),
        ServerDisconnectedError(),
        _make_client_response_error(429),
        _make_client_response_error(503),
    ],
)
async def test_should_retry_on_exceptions(
    raised: Exception, test_app_config: AppConfig
) -> None:
    # Given
    client = MockFailingClient(test_app_config.gotenberg_url, raised)
    # When
    async with client:
        converted = await client.convert_doc_to_pdf(b"convert_me", ext=".txt")
    # Then
    assert converted == b"success"


@pytest.mark.parametrize(
    "raised",
    [
        ClientConnectionError(),
        ClientOSError(),
        _make_client_response_error(401),
        ValueError("error"),
    ],
)
async def test_should_fail_on_unhandled_exceptions(
    raised: Exception, test_app_config: AppConfig
) -> None:
    # Given
    client = MockFailingClient(test_app_config.gotenberg_url, raised)
    # When
    async with client:
        with pytest.raises(raised.__class__):
            await client.convert_doc_to_pdf(b"convert_me", ext=".txt")
