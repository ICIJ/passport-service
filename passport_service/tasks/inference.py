import logging
from collections.abc import AsyncGenerator, Sequence
from pathlib import Path

import onnxruntime

from passport_service.core.object_detection import detect_passports
from passport_service.objects import DetectionRequest, PassportDetection

logger = logging.getLogger(__name__)


async def passport_detection_task(
    inputs: Sequence[DetectionRequest],
    inference_sess: onnxruntime.InferenceSession,
    passport_class_name: str,
    *,
    data_dir: Path,
    work_dir: Path,
    batch_size: int,
    read_mrz: bool,
    country_codes: list[str],
) -> AsyncGenerator[PassportDetection, None]:
    inputs = [r.relative_to(data_dir=data_dir, work_dir=work_dir) for r in inputs]
    logger.info("Running passport detection on %s docs...", len(inputs))
    classes = [passport_class_name]
    async for d in detect_passports(
        inputs=inputs,
        batch_size=batch_size,
        sess=inference_sess,
        classes=classes,
        read_mrz=read_mrz,
        country_codes=country_codes,
    ):
        yield d.relative_to(data_dir=data_dir)
