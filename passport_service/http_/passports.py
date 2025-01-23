import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from icij_worker import Task
from icij_worker.exceptions import TaskAlreadyQueued, TaskQueueIsFull
from starlette.responses import Response

from passport_service.constants import (
    CREATE_PREPROCESSING_TASKS_TASK,
    DETECT_PASSPORTS_TASKS,
    PREPROCESS_DOCS_TASK,
)
from passport_service.http_ import PASSPORTS_TAG
from passport_service.http_.dependencies import lifespan_task_manager
from passport_service.objects import (
    PassportDetectionRequest,
    PreprocessingRequest,
    PreprocessingTaskRequest,
    generate_task_id,
)

logger = logging.getLogger(__name__)

_PREPROCESS_TASK_NAME = "preprocess-docs"


def passports_router() -> APIRouter:
    router = APIRouter(tags=[PASSPORTS_TAG])

    @router.post("/passports/preprocessing-tasks", response_model=str)
    async def _create_preprocessing_tasks(request: PreprocessingTaskRequest) -> str:
        args = request.dict(by_alias=True)
        return await _create_task(args, CREATE_PREPROCESSING_TASKS_TASK)

    @router.post("/passports/preprocessing", response_model=str)
    async def _preprocess_documents(request: PreprocessingRequest) -> str:
        args = request.dict(by_alias=True)
        return await _create_task(args, PREPROCESS_DOCS_TASK)

    @router.post("/passports/detection", response_model=str)
    async def _detect_passports(request: PassportDetectionRequest) -> str:
        args = request.dict(by_alias=True)
        return await _create_task(args, DETECT_PASSPORTS_TASKS)

    return router


async def _create_task(args: Any, task_name: str) -> Response:
    task_manager = lifespan_task_manager()
    task_id = generate_task_id(task_name)
    task = Task.create(task_id=task_id, task_name=task_name, args=args)
    is_new = await task_manager.save_task(task)
    try:
        await task_manager.enqueue(task)
    except TaskAlreadyQueued:
        pass
    except TaskQueueIsFull as e:
        raise HTTPException(429, detail="Too Many Requests") from e
    status = 201 if is_new else 200
    return Response(task.id, status_code=status)
