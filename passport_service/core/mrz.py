from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

import numpy as np
from imageio import imwrite
from passporteye.mrz.image import (
    BooneTransform,
    BoxToMRZ,
    ExtractAllBoxes,
    FindFirstValidMRZ,
    Loader,
    MRZBoxLocator,
    MRZPipeline,
    Scaler,
    TryOtherMaxWidth,
)
from passporteye.mrz.text import MRZ
from passporteye.util.geometry import RotatedBox
from passporteye.util.pipeline import Pipeline
from pytesseract import pytesseract
from skimage import morphology, transform

from passport_service import DATA_DIR
from passport_service.typing_ import PassportEyeMRZ


class BoxToMRZFixed(BoxToMRZ):
    # Copied from BoxToMRZ

    __provides__ = ["roi", "text", "mrz"]
    __depends__ = ["box", "img", "img_small", "scale_factor"]

    def __init__(
        self,
        extra_cmdline_params: str = "",
        mrz_ocr: Callable | None = None,
        *,
        use_original_image: bool = True,
    ):
        super().__init__(use_original_image, extra_cmdline_params)
        if mrz_ocr is None:
            mrz_ocr = tesseract_mrz_ocr
        self.mrz_ocr = mrz_ocr

    def __call__(
        self,
        box: RotatedBox,
        img: np.ndarray,
        img_small: np.ndarray,
        scale_factor: float,
    ) -> tuple[np.ndarray, str, PassportEyeMRZ]:
        img = img if self.use_original_image else img_small
        scale = 1.0 / scale_factor if self.use_original_image else 1.0
        roi = box.extract_from_image(img, scale)
        text = self.mrz_ocr(roi, extra_cmdline_params=self.extra_cmdline_params)

        if ">>" in text or (">" in text and "<" not in text):
            # Most probably we need to reverse the ROI
            roi = roi[::-1, ::-1]
            text = self.mrz_ocr(roi, extra_cmdline_params=self.extra_cmdline_params)

        if "<" not in text:
            # Assume this is unrecoverable and stop here
            # (TODO: this may be premature, although it saves time on useless stuff)
            return roi, text, MRZ.from_ocr(text)

        mrz = MRZ.from_ocr(text)
        mrz.aux["method"] = "direct"

        # Now try improving the result via hacks
        if not mrz.valid:
            text, mrz = self._try_larger_image(roi, text, mrz)

        # Sometimes the filter used for enlargement is important!
        if not mrz.valid:
            text, mrz = self._try_larger_image(roi, text, mrz, 1)

        if not mrz.valid:
            text, mrz = self._try_black_tophat(roi, text, mrz)

        return roi, text, mrz

    def _try_larger_image(
        self,
        roi: np.ndarray,
        cur_text: str,
        cur_mrz: PassportEyeMRZ,
        filter_order: int = 3,
    ) -> tuple[str, MRZ]:
        # Attempts to improve the OCR result by scaling the image. If the new mrz is
        # better, returns it, otherwise returns the old mrz.
        if roi.shape[1] <= 700:
            scale_by = int(1050.0 / roi.shape[1] + 0.5)
            roi_lg = transform.rescale(
                roi,
                scale_by,
                order=filter_order,
                mode="constant",
                anti_aliasing=True,
            )
            new_text = self.mrz_ocr(
                roi_lg, extra_cmdline_params=self.extra_cmdline_params
            )
            new_mrz = MRZ.from_ocr(new_text)
            new_mrz.aux["method"] = f"rescaled({filter_order:d})"
            if new_mrz.valid_score > cur_mrz.valid_score:
                cur_mrz = new_mrz
                cur_text = new_text
        return cur_text, cur_mrz

    def _try_black_tophat(
        self, roi: np.ndarray, cur_text: str, cur_mrz: PassportEyeMRZ
    ) -> tuple[str, MRZ]:
        roi_b = morphology.black_tophat(roi, morphology.disk(5))
        # There are some examples where this line basically hangs for an undetermined
        # amount of time.
        new_text = self.mrz_ocr(roi_b, extra_cmdline_params=self.extra_cmdline_params)
        new_mrz = MRZ.from_ocr(new_text)
        if new_mrz.valid_score > cur_mrz.valid_score:
            new_mrz.aux["method"] = "black_tophat"
            cur_text, cur_mrz = new_text, new_mrz

        new_text, new_mrz = self._try_larger_image(roi_b, cur_text, cur_mrz)
        if new_mrz.valid_score > cur_mrz.valid_score:
            new_mrz.aux["method"] = "black_tophat(rescaled(3))"
            cur_text, cur_mrz = new_text, new_mrz

        return cur_text, cur_mrz


class FindFirstValidMRZFixed(FindFirstValidMRZ):
    def __init__(
        self,
        extra_cmdline_params: str = "",
        *,
        use_original_image: bool = True,
    ):
        super().__init__(use_original_image, extra_cmdline_params)
        self.box_to_mrz = BoxToMRZFixed(
            use_original_image=use_original_image,
            extra_cmdline_params=extra_cmdline_params,
        )


class MRZPipelineFixed(MRZPipeline):
    def __init__(self, file: str | BytesIO, extra_cmdline_params: str = ""):
        super(MRZPipeline, self).__init__()
        self.version = "1.0"
        self.file = file
        self.add_component("loader", Loader(file))
        self.add_component("scaler", Scaler())
        self.add_component("boone", BooneTransform())
        self.add_component("box_locator", MRZBoxLocator())
        self.add_component(
            "mrz", FindFirstValidMRZFixed(extra_cmdline_params=extra_cmdline_params)
        )
        self.add_component("other_max_width", TryOtherMaxWidth())
        self.add_component("extractor", ExtractAllBoxes())


class MockProvider:
    __depends__ = ["img"]
    __provides__ = [
        "img_small",
        # We artificially provided this factor even if we don't scale as it's needed by
        # the last pipeline steps
        "scale_factor",
    ]

    def __call__(self, img: np.ndarray):
        img_small = img
        scale_factor = 1.0
        return img_small, scale_factor


class PassThroughBoxLocator:
    # Return the full image as a MRZ box
    __depends__ = ["img"]
    __provides__ = [
        # We artificially provided this factor even if we don't scale as it's needed by
        # the last pipeline steps
        "boxes",
    ]

    def __call__(self, img: np.ndarray):
        # Return a single box encompassing the full image
        # TODO: we assume the image was provided correctly rotated

        height, width = img.shape
        center = (height / 2, width / 2)
        box = RotatedBox(center, width, height, np.pi / 2)
        boxes = [box]
        return boxes


class YoloMRZPipeline(Pipeline):
    # MRZ pipeline which skips MRZ
    def __init__(self, file: str | Path, extra_cmdline_params: str = ""):
        super().__init__()
        self.version = "1.0"
        if isinstance(file, Path):
            file = str(file)
        self.file = file
        self.add_component("loader", Loader(file))
        self.add_component("provided", MockProvider())
        # We remove the boone transform since it was needed for box detection
        # self.add_component("boone", BooneTransform())
        self.add_component("box_locator", PassThroughBoxLocator())
        # We keep this to avoid bugs
        self.add_component(
            "mrz", FindFirstValidMRZ(extra_cmdline_params=extra_cmdline_params)
        )
        # We keep this to avoid bugs
        self.add_component("other_max_width", TryOtherMaxWidth())
        self.add_component("extractor", ExtractAllBoxes())

    @property
    def result(self) -> MRZ:
        return self["mrz_final"]


def read_mrz_yolo(
    file: BytesIO | Path | str,
    extra_cmdline_params: str = "",
    *,
    save_roi: bool = False,
) -> MRZ:
    p = YoloMRZPipeline(file, extra_cmdline_params)
    mrz = p.result
    if mrz is not None and save_roi:
        mrz.aux["roi"] = p["roi"]
    return mrz


def read_passport_file_mrz(
    file: str | BytesIO, extra_cmdline_params: str = "", *, save_roi: bool = False
) -> PassportEyeMRZ | None:
    p = MRZPipelineFixed(file, extra_cmdline_params)
    mrz = p.result
    if mrz is not None and save_roi:
        mrz.aux["roi"] = p["roi"]
    return mrz


def tesseract_mrz_ocr(img: np.ndarray, extra_cmdline_params: str = "") -> str:
    if img is None or img.shape[-1] == 0:  # Issue #34
        return ""
    input_file = NamedTemporaryFile(prefix="tess_", suffix=".bmp")  # noqa: SIM115
    output_file = NamedTemporaryFile(prefix="tess_output_")  # noqa: SIM115
    try:
        with input_file as i_f, output_file as o_f:
            # Prevent annoying warning about lossy conversion to uint8
            if (
                str(img.dtype).startswith("float")
                and np.nanmin(img) >= 0
                and np.nanmax(img) <= 1
            ):
                img = img.astype(np.float64) * (np.power(2.0, 8) - 1) + 0.499999999
                img = img.astype(np.uint8)
            imwrite(i_f.name, img)
            config = f"--psm 6 --tessdata-dir {DATA_DIR} {extra_cmdline_params}"
            pytesseract.run_tesseract(
                i_f.name, str(o_f.name), "txt", lang="mrz", config=config
            )
            mrz = Path(o_f.name).with_suffix(".txt").read_text().strip()
            return mrz
    finally:
        pytesseract.cleanup(input_file.name)
        pytesseract.cleanup(output_file.name)
