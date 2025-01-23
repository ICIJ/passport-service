import logging
from pathlib import Path

from icij_worker import AsyncApp
from pydantic import parse_obj_as

from passport_service.constants import (
    CREATE_PREPROCESSING_TASKS_TASK,
    DETECT_PASSPORTS_TASKS,
    INFERENCE_GROUP,
    PREPROCESS_DOCS_TASK,
    PREPROCESSING_GROUP,
    Colorspace,
)
from passport_service.objects import DetectionRequest, DocMetadata, as_doc_metadata
from passport_service.tasks.dependencies import (
    APP_LIFESPAN_DEPS,
    lifespan_config,
    lifespan_gotenberg_client,
    lifespan_pool_executor,
    lifespan_task_client,
)

logger = logging.getLogger(__name__)

app = AsyncApp("passport-detection", dependencies=APP_LIFESPAN_DEPS)


@app.task(name=CREATE_PREPROCESSING_TASKS_TASK, group=PREPROCESSING_GROUP)
async def create_preprocessing_tasks(
    docs: str | list[dict],
    batch_size: int = 64,
    *,
    detection_args: dict,
) -> list[str]:
    from passport_service.tasks.preprocessing import (
        create_preprocessing_tasks as create_preprocessing_tasks_,
    )

    if isinstance(docs, str):
        config = lifespan_config()
        data_dir = config.data_dir
        logger.debug("exploring files in %s", data_dir.absolute())
        docs_dir = Path(config.data_dir) / docs
        docs = as_doc_metadata(docs_dir, data_dir=data_dir)
        logger.debug("found %s and more ...", docs[:10])
    else:
        docs = parse_obj_as(list[DocMetadata], docs)
    task_client = lifespan_task_client()
    tasks_ids = await create_preprocessing_tasks_(
        docs, task_client, batch_size, detection_args
    )
    return tasks_ids


@app.task(name=PREPROCESS_DOCS_TASK, group=PREPROCESSING_GROUP)
async def preprocess_docs(docs: list[dict], detection_args: dict) -> dict:
    from passport_service.tasks.preprocessing import preprocess_docs_task

    config = lifespan_config()
    data_dir = config.data_dir
    work_dir = config.work_dir

    docs = parse_obj_as(list[DocMetadata], docs)
    executor = lifespan_pool_executor()
    gotenberg_client = lifespan_gotenberg_client()
    task_client = lifespan_task_client()
    colorspace = Colorspace.RGB
    pdf_conversion_concurrency = config.pdf_conversion_concurrency
    response = await preprocess_docs_task(
        docs,
        detection_args,
        work_dir=work_dir,
        data_dir=data_dir,
        colorspace=colorspace,
        executor=executor,
        gotenberg_client=gotenberg_client,
        task_client=task_client,
        preprocessing_batch_size=config.preprocessing_batch_size,
        pdf_conversion_concurrency=pdf_conversion_concurrency,
    )
    response = response.dict(by_alias=True)
    return response


@app.task(name=DETECT_PASSPORTS_TASKS, group=INFERENCE_GROUP)
async def detect_passports(
    inputs: list[dict], model_path: Path, *, read_mrz: bool = True
) -> list[dict]:
    from passport_service.core.object_detection import inference_session
    from passport_service.tasks.inference import passport_detection_task

    inputs = parse_obj_as(list[DetectionRequest], inputs)
    config = lifespan_config()
    work_dir = config.work_dir
    data_dir = config.data_dir
    model_path = work_dir / model_path
    logger.info("loading model %s", model_path)
    with inference_session(model_path) as sess:
        countries = config.countries
        batch_size = config.inference_batch_size
        passport_class_name = config.passport_label
        detections = passport_detection_task(
            inputs,
            sess,
            passport_class_name=passport_class_name,
            data_dir=data_dir,
            work_dir=work_dir,
            country_codes=countries,
            batch_size=batch_size,
            read_mrz=read_mrz,
        )
        detections = [d.dict(by_alias=True) async for d in detections]
        return detections
