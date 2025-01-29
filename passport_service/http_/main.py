import importlib

from fastapi import APIRouter
from starlette.responses import Response
from starlette.status import HTTP_200_OK, HTTP_503_SERVICE_UNAVAILABLE

from passport_service.http_.dependencies import lifespan_task_manager
from passport_service.http_.doc import OTHER_TAG


def main_router() -> APIRouter:
    router = APIRouter(tags=[OTHER_TAG])

    # TODO: add a ping route

    @router.get("/version")
    def version() -> Response:
        import passport_service

        package_version = importlib.metadata.version(passport_service.__name__)

        return Response(content=package_version, media_type="text/plain")

    @router.get("/health")
    async def health(response: Response) -> dict[str, bool]:
        # TODO: add Gotenberg client status ?
        task_manager = lifespan_task_manager()
        health = await task_manager.get_health()
        is_healthy = all(health.values())
        response.status_code = (
            HTTP_200_OK if is_healthy else HTTP_503_SERVICE_UNAVAILABLE
        )
        return health

    return router
