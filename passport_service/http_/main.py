import importlib

from fastapi import APIRouter
from starlette.responses import Response

from passport_service.http_.doc import OTHER_TAG


def main_router() -> APIRouter:
    router = APIRouter(tags=[OTHER_TAG])

    # TODO: add a ping route

    @router.get("/version")
    def version() -> Response:
        import passport_service

        package_version = importlib.metadata.version(passport_service.__name__)

        return Response(content=package_version, media_type="text/plain")

    return router
