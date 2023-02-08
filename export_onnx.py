import argparse
import torch
from src.net import Net


def export_onnx(model_path):
    model = Net()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    # export onnx
    dummy_input = torch.randn(1, 1, 32, 32)
    torch.onnx.export(model, dummy_input, model_path.split('.')[0] + '.onnx', input_names=['input'],
                      output_names=['output'], verbose=True,
                      opset_version=11)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    args = parser.parse_args()

    model_path = args.model

    export_onnx(model_path)
