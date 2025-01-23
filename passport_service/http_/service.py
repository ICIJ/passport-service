import logging
from collections.abc import Iterable

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from icij_common.fastapi_utils import (
    http_exception_handler,
    internal_exception_handler,
    request_validation_error_handler,
)
from starlette.exceptions import HTTPException as StarletteHTTPException

from passport_service.config import HttpServiceConfig
from passport_service.http_ import OTHER_TAG, PASSPORTS_TAG, TASKS_TAG
from passport_service.http_.dependencies import run_http_service_deps
from passport_service.http_.main import main_router
from passport_service.http_.passports import passports_router
from passport_service.http_.tasks import tasks_router

INTERNAL_SERVER_ERROR = "Internal Server Error"
_REQUEST_VALIDATION_ERROR = "Request Validation Error"

logger = logging.getLogger(__name__)


def _make_open_api_tags(tags: Iterable[str]) -> list[dict]:
    return [{"name": t} for t in tags]


def create_service(config: HttpServiceConfig | None) -> FastAPI:
    app_title = "Passport detection service 🕵️"
    app = FastAPI(
        title=app_title,
        openapi_tags=_make_open_api_tags([PASSPORTS_TAG, TASKS_TAG, OTHER_TAG]),
        lifespan=run_http_service_deps,
    )
    app.state.config = config
    app.add_exception_handler(RequestValidationError, request_validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, internal_exception_handler)
    app.include_router(main_router())
    app.include_router(passports_router())
    app.include_router(tasks_router())
    return app
