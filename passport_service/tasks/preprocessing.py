import logging
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

from passport_service.constants import (
    DETECT_PASSPORTS_TASKS,
    PREPROCESS_DOCS_TASK,
    Colorspace,
)
from passport_service.core.pdf_conversion import GotenbergClient
from passport_service.core.preprocessing import preprocess_docs
from passport_service.http_ import TaskClient
from passport_service.objects import DocMetadata, ProcessingResponse
from passport_service.utils import batches

logger = logging.getLogger(__name__)


async def create_preprocessing_tasks(
    docs_metadata: list[DocMetadata],
    task_client: TaskClient,
    batch_size: int,
    detection_args: dict,
) -> list[str]:
    task_batches = batches(docs_metadata, batch_size=batch_size)
    logger.info("spawning preprocessing tasks...")
    task_ids = []
    for batch in task_batches:
        args = {"docs": list(batch), "detection_args": detection_args}
        task_id = await task_client.create_task(PREPROCESS_DOCS_TASK, args)
        task_ids.append(task_id)
    logger.info("created %s preprocessing tasks !", len(task_ids))
    return task_ids


async def preprocess_docs_task(
    docs_metadata: list[DocMetadata],
    detection_args: dict,
    *,
    work_dir: Path,
    data_dir: Path,
    colorspace: Colorspace = Colorspace.RGB,
    executor: ProcessPoolExecutor,
    gotenberg_client: GotenbergClient,
    task_client: TaskClient,
    preprocessing_batch_size: int,
    pdf_conversion_concurrency: int,
) -> ProcessingResponse:
    docs_metadata = [doc.relative_to(data_dir=data_dir) for doc in docs_metadata]
    reports = [
        report.relative_to(data_dir=data_dir, work_dir=work_dir)
        async for report in preprocess_docs(
            docs_metadata,
            work_dir,
            executor,
            preprocessing_batch_size=preprocessing_batch_size,
            gotenberg_client=gotenberg_client,
            pdf_conversion_concurrency=pdf_conversion_concurrency,
            colorspace=colorspace,
        )
    ]
    args = deepcopy(detection_args)
    args["inputs"] = [r.dict(by_alias=True) for r in reports]
    task_id = await task_client.create_task(DETECT_PASSPORTS_TASKS, args)
    response = ProcessingResponse(detection_task_id=task_id, reports=reports)
    return response
