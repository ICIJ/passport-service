from pathlib import Path

import cv2

from passport_service.core.object_detection import (
    detect_objects,
    inference_session,
    preprocess_image,
    read_passport_mrz,
)
from passport_service.objects import MRZ, ObjectDetection
from tests import TEST_DATA_DIR


def test_detect_objects(test_model_v0: Path) -> None:  # noqa: ARG001
    # Given
    classes = ["passport"]
    with inference_session(test_model_v0) as sess:
        images_dir = TEST_DATA_DIR.joinpath("passports")
        image_paths = [
            images_dir / "not_a_passport.jpg",
            images_dir / "passport.png",
        ]
        inputs = [preprocess_image(cv2.imread(str(p))) for p in image_paths]
        inputs = [(img, scale) for _, img, scale in inputs]

        # When
        detections = list(detect_objects(sess, inputs, classes))

        # Then
        assert len(detections) == 2
        not_passport_detection = detections[0]
        assert not not_passport_detection
        passport_detection = detections[1]
        assert len(passport_detection) == 2  # 2 passport pages
        assert all(detection.class_id == "passport" for detection in passport_detection)


def test_read_passport_mrz() -> None:
    # Given
    country_codes = ["EOL", "FRA"]
    images_dir = TEST_DATA_DIR.joinpath("passports")
    im_path = images_dir / "passport.png"
    page, _, scale = preprocess_image(cv2.imread(im_path))
    box = (4.0765380859375, 240.26063537597656, 541.396728515625, 395.6768493652344)
    passport_detection = ObjectDetection(
        class_id="passport", confidence=1.0, box=box, scale=scale
    )
    # When
    mrz = read_passport_mrz(page, passport_detection, country_codes=country_codes)
    # Then
    assert isinstance(mrz, MRZ)
    assert mrz.country == "EOL"
    assert isinstance(mrz.metadata, dict)
