import sys
from pathlib import Path

import onnx
from onnx.tools.update_model_dims import update_inputs_outputs_dims


def make_dynamic_model(model_path: Path, *, output_path: Path) -> None:
    model = onnx.load(model_path)
    input_dims = {
        i.name: [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]
        for i in model.graph.input
    }
    input_dims["images"][0] = "batch"
    output_dims = {
        i.name: [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]
        for i in model.graph.output
    }
    output_dims["output0"][0] = "batch"
    model = update_inputs_outputs_dims(model, input_dims, output_dims)
    onnx.save(model, output_path)


if __name__ == "__main__":
    model_path = sys.argv[1]
    model_path = Path(model_path)
    output_path = sys.argv[2]
    output_path = Path(output_path)
    make_dynamic_model(model_path, output_path=output_path)
