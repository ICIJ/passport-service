import gc
import logging
import re
from collections import defaultdict
from collections.abc import AsyncGenerator, Generator, Sequence
from contextlib import contextmanager
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import cast

import cv2
import Levenshtein as Lev
import numpy as np
import onnxruntime as rt
import pycountry
from cv2.dnn import NMSBoxes, blobFromImage
from cv2.typing import MatLike
from icij_worker.typing_ import RateProgress
from icij_worker.utils.progress import to_raw_progress
from onnxruntime import SessionOptions
from PIL.Image import fromarray

from ..constants import COLOR_LUT, PIL_PNG, Colorspace
from ..objects import (
    MRZ,
    DetectionRequest,
    DocPageDetection,
    ObjectDetection,
    Passport,
    PassportDetection,
)
from ..typing_ import BoxLocation, PassportEyeMRZ
from .mrz import read_passport_file_mrz

logger = logging.getLogger(__name__)

DEFAULT_DETECTION_THRESHOLD = 0.05
DEFAULT_NMS_THRESHOLD = 0.45
DEFAULT_NMS_SCORE_THRESHOLD = 0.25
DEFAULT_NMS_ETA = 0.5

_DEU_PATTERN_0 = re.compile(r"[0-9]D.")
_DEU_PATTERN_1 = re.compile(r"D<<")


async def detect_passports(
    inputs: Sequence[DetectionRequest],
    batch_size: int,
    sess: rt.InferenceSession,
    *,
    classes: Sequence[str],
    read_mrz: bool,
    country_codes: list[str],
    progress: RateProgress | None = None,
) -> AsyncGenerator[PassportDetection, None]:
    n_inputs = len(inputs)
    errors = [i for i in inputs if i.error]
    for e in errors:
        detection = PassportDetection(
            doc_path=e.doc_path, doc_pages=e.pages, error=e.error
        )
        yield detection
    inputs = [i for i in inputs if not i.error]
    n_dropped = n_inputs - len(inputs)
    if n_dropped:
        logger.info("dropped %s documents due to preprocessing error !", n_dropped)
    n_batches = n_inputs // batch_size
    if progress is not None:
        progress = to_raw_progress(progress, n_batches)
    pages_it = _batched_pages_it(inputs, batch_size)
    buffer = defaultdict(list)
    n_pages = {i: len(req.pages) for i, req in enumerate(inputs)}
    for n_batch, (doc_idxs, page_paths) in enumerate(pages_it):
        preprocessed = (
            preprocess_image(cv2.imread(str(page_path))) for page_path in page_paths
        )
        pages, blobs, scales = zip(*preprocessed)
        batch = list(zip(blobs, scales))
        passports_per_page = zip(pages, detect_objects(sess, batch, classes))
        if read_mrz:
            passports_per_page = (
                [
                    _add_mrzs(page, passport, country_codes=country_codes)
                    for passport in passports
                ]
                for page, passports in passports_per_page
            )
        doc_paths = [inputs[doc_ix].doc_path for doc_ix in doc_idxs]
        buffer, detections = _update_buffer(
            buffer, dict(zip(doc_idxs, passports_per_page)), n_pages, doc_paths
        )
        for detection in detections:
            yield detection
        if progress is not None:
            await progress(n_batch)
    if buffer:
        msg = "inconsistent state: buffer was not emptied"
        msg += f"\nbuffer: {buffer}\nn_pages: {n_pages}"
        raise ValueError(msg)


def _batched_pages_it(
    inputs: Sequence[DetectionRequest], batch_size: int
) -> Generator[tuple[list[int], list[Path]], None, None]:
    batch = []
    for doc_i, req in enumerate(inputs):
        for page in req.pages:
            batch.append((doc_i, page))
            if len(batch) == batch_size:
                doc_ixs, pages = zip(*batch)
                yield list(doc_ixs), list(pages)
                batch.clear()
    if batch:
        doc_ixs, pages = zip(*batch)
        yield list(doc_ixs), list(pages)
        batch.clear()


def _update_buffer(
    buffer: dict[int, list[list[Passport]]],
    passports_per_page: dict[int, list[ObjectDetection]],
    n_pages: dict[int, int],
    doc_paths: Sequence[Path],
) -> tuple[dict[int, list[list[Passport]]], list[PassportDetection]]:
    detections = []
    for doc_ix, page_detections in passports_per_page.items():
        page_passports = []
        for page_detection in page_detections:
            passport = page_detection
            if not isinstance(passport, Passport):  # no mrz detection
                passport = Passport.from_detection(passport)
            page_passports.append(passport)
        buffer[doc_ix].append(page_passports)
    to_pop = []
    for doc_ix, pages in buffer.items():
        if len(pages) == n_pages[doc_ix]:
            doc_path = doc_paths[doc_ix]
            to_pop.append(doc_ix)
            doc_passports = buffer[doc_ix]
            doc_passports = [
                DocPageDetection(page=page_i, passports=passports)
                for page_i, passports in enumerate(doc_passports)
                if passports
            ]
            detection = PassportDetection(doc_path=doc_path, doc_pages=doc_passports)
            detections.append(detection)
    for ix in to_pop:
        buffer.pop(ix)
    return buffer, detections


_DEFAULT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


@contextmanager
def inference_session(path: Path) -> Generator[rt.InferenceSession, None, None]:
    sess = None
    try:
        sess_options = SessionOptions()
        sess_options.log_severity_level = 0
        sess = rt.InferenceSession(
            path, providers=_DEFAULT_PROVIDERS, sess_options=sess_options
        )
        yield sess
    finally:
        if sess is not None:
            del sess
            gc.collect()


def preprocess_image(
    image: MatLike, target_size: float = 640
) -> tuple[np.ndarray, MatLike, float]:
    [height, width, _] = image.shape
    length = max((height, width))
    page = np.zeros((length, length, 3), np.uint8)
    page[0:height, 0:width] = image
    scale = length / target_size
    size = (target_size, target_size)
    blob = blobFromImage(page, scalefactor=1 / 255, size=size, swapRB=True)
    return page, blob, scale


def detect_objects(
    sess: rt.InferenceSession,
    inputs: Sequence[tuple[MatLike, float]],
    classes: Sequence[str],
    *,
    detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
    nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    nms_score_threshold: float = DEFAULT_NMS_SCORE_THRESHOLD,
    nms_eta: float = DEFAULT_NMS_ETA,
) -> Generator[list[ObjectDetection], None, None]:
    if not inputs:
        return
    blobs, scales = zip(*inputs)
    blobs = np.concatenate(blobs)
    scales = list(scales)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    model_inputs = {input_name: blobs.astype(np.float32)}
    outputs = sess.run(  # [batch_size, n_classes + dim_box, max_boxes = 8400]
        [label_name], model_inputs
    )[0]
    outputs = np.array(outputs, dtype=np.float32).reshape(
        (-1, outputs.shape[-2], outputs.shape[-1])
    )
    for output, scale in zip(outputs, scales):
        detection = detections_from_nn_output(
            output,
            classes,
            scale=scale,
            detection_threshold=detection_threshold,
            nms_threshold=nms_threshold,
            nms_score_threshold=nms_score_threshold,
            nms_eta=nms_eta,
        )
        yield detection


def to_pillow_box(box_loc: BoxLocation) -> BoxLocation:
    center_x, center_y, width, height = box_loc
    upper_left_x = float(center_x - 0.5 * width)
    upper_left_y = float(center_y - 0.5 * height)
    return upper_left_x, upper_left_y, width, height


def detections_from_nn_output(
    detection_output: np.ndarray,
    classes: Sequence[str],
    *,
    scale: float,
    detection_threshold: float,
    nms_score_threshold: float,
    nms_threshold: float,
    nms_eta: float,
) -> list[ObjectDetection]:
    detection_output = detection_output.transpose()
    detections = []
    class_confidences = detection_output[:, 4:]
    max_confidences_ix = np.argmax(class_confidences, axis=-1)
    detected = (class_confidences.max(axis=-1) >= detection_threshold).nonzero()[0]
    for detected_i in detected:
        # dim_box = box_coordinates + scores = 4 + ....
        class_ix = max_confidences_ix[detected_i]
        confidence = class_confidences[detected_i, class_ix]
        box_loc = tuple(detection_output[detected_i, :4])
        box_loc = cast(BoxLocation, box_loc)
        box = to_pillow_box(box_loc)
        detections.append((box, class_ix, confidence))
    if not detections:
        return []
    boxes, class_ixs, scores = zip(*detections)
    boxes = list(boxes)
    scores = list(scores)
    class_ixs = list(class_ixs)
    s = NMSBoxes(boxes, scores, nms_score_threshold, nms_threshold, nms_eta)
    nms_boxes = s
    detections = []
    for box_i in nms_boxes:
        box = boxes[box_i]
        confidence = scores[box_i]
        class_id = classes[class_ixs[box_i]]
        detection = ObjectDetection(
            class_id=class_id, confidence=confidence, box=box, scale=scale
        )
        detections.append(detection)
    return detections


def read_passport_mrz(
    page: np.ndarray,
    passport: ObjectDetection,
    country_codes: list[str],
    mrz_extras_cmdline: str = "",
) -> MRZ | None:
    # TODO: use special Tesseract MRZ language ?
    passport_image = _read_image_box(page, passport.box, scale=passport.scale)
    with BytesIO() as f:
        fromarray(passport_image, mode=COLOR_LUT[Colorspace.RGB]).save(
            f, format=PIL_PNG
        )
        # TODO: use Yolo for MRZ detection to avoid re-doing box detection ?
        maybe_mrz = read_passport_file_mrz(f, extra_cmdline_params=mrz_extras_cmdline)
    if maybe_mrz is None:
        return None
    mrz_fn = partial(_get_mrz_country, countries=country_codes)
    mrz = MRZ.from_passport_eye(maybe_mrz, mrz_to_country=mrz_fn)
    return mrz


def _add_mrzs(
    page: np.array, passport: ObjectDetection, country_codes: list[str]
) -> Passport:
    mrz = read_passport_mrz(page, passport, country_codes=country_codes)
    passport = Passport.from_detection(passport, mrz)
    return passport


def _is_match(pattern: re.Pattern, string: str) -> bool:
    return pattern.fullmatch(string) is not None


def _get_mrz_country(mrz: PassportEyeMRZ, countries: Sequence[str]) -> str | None:
    country = None
    if mrz is not None:
        mrz_dict = mrz.to_dict()
        country = mrz_dict.get("country")
        if country not in countries:
            country = _closest_word(country, countries)
        if country is not None:
            normalized = pycountry.countries.get(alpha_3=country)
            if normalized is not None:
                return normalized.name
    return country


def _closest_word(word: str, candidates: Sequence[str]) -> str | None:
    closest = None
    closest_distance = None
    if _is_match(_DEU_PATTERN_0, word) or _is_match(_DEU_PATTERN_1, word):
        word = "DEU"
        return word
    for c in candidates:
        distance = Lev.distance(c, word)
        if closest_distance is None or distance < closest_distance:
            closest_distance = distance
            closest = word

    return closest


def _read_image_box(page: np.ndarray, box: BoxLocation, scale: float) -> np.ndarray:
    # Copied from passporteye Loader
    box = tuple(int(coordinate * scale) for coordinate in box)
    box = cast(BoxLocation, box)
    boxed = _crop_image(page, box)
    return boxed


def _crop_image(img: np.ndarray, box: BoxLocation) -> np.ndarray:
    left, upper, width, height = box
    right = left + width
    bottom = upper + height
    border = (left, upper, right, bottom)
    boxed = fromarray(img).crop(border)
    boxed = np.array(boxed)
    return boxed
