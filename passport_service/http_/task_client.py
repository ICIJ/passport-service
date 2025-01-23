import logging
import uuid
from typing import Any

from icij_common.pydantic_utils import jsonable_encoder
from icij_worker.utils.http import AiohttpClient

logger = logging.getLogger(__name__)


class TaskClient(AiohttpClient):
    async def create_task(
        self, name: str, args: dict[str, Any], *, id_: str | None = None
    ) -> str:
        if id_ is None:
            id_ = _generate_task_id(name)
        data = {"name": name, "args": jsonable_encoder(args)}
        url = f"/tasks/{id_}"
        async with self._put(url, json=data) as res:
            return await res.text()


def _generate_task_id(task_name: str) -> str:
    return f"{task_name}-{uuid.uuid4()}"
