import argparse

import torch

from src.net import Net, MLP


def export_onnx(model_path, arch="net", output_path=None):
    arch = arch.lower()
    if arch == "net":
        model = Net()
        dummy_input = torch.randn(1, 1, 32, 32)
    elif arch == "mlp":
        model = MLP()
        dummy_input = torch.randn(1, 1, 28, 28)
    else:
        raise ValueError("arch must be in [net | mlp]")

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    if output_path is None:
        output_path = model_path.split(".")[0] + ".onnx"

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        verbose=True,
        opset_version=11,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("--arch", type=str, default="net", help="model arch. [net | mlp]")
    parser.add_argument("-o", "--output", type=str, required=False, default=None, help="output onnx path")
    args = parser.parse_args()

    model_path = args.model

    export_onnx(model_path, arch=args.arch, output_path=args.output)
