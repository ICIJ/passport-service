import shutil
import sys
from pathlib import Path


def convert_torch_to_dynamic_onnx(model_path: Path, *, output_path: Path) -> None:
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(model_path)
    exported = model.export(format="onnx", dynamic=True)
    shutil.move(exported, output_path)


if __name__ == "__main__":
    model_path = sys.argv[1]
    model_path = Path(model_path)
    output_path = sys.argv[2]
    output_path = Path(output_path)
    convert_torch_to_dynamic_onnx(model_path, output_path=output_path)
