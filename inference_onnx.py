import onnxruntime
import cv2 as cv
import numpy as np

from time import time
import argparse


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_test_transform():
    from torchvision import transforms
    return transforms.ToTensor()


def inference_mnist(model_path, img_path):
    session = onnxruntime.InferenceSession(model_path)
    src = cv.imread(img_path, 0)
    w, h = 32, 32
    resized_img = cv.resize(src, (w, h))
    img = get_test_transform()(resized_img)
    img = img.unsqueeze_(0)

    inputs = {session.get_inputs()[0].name: to_numpy(img)}
    start_time = time()
    outs = session.run(None, inputs)
    end_time = time()
    print(f"Inference time: {end_time - start_time:.6f} ms")
    print(f"The prediction digit: {np.argmax(outs)}")

    res_img = cv.resize(src, None, fx=10, fy=10)
    rgb_img = cv.cvtColor(res_img, cv.COLOR_GRAY2BGR)
    cv.putText(rgb_img, "Prediction:" + str(np.argmax(outs)), (0, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv.imshow("Result", rgb_img)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-i', '--img', type=str, required=True)
    args = parser.parse_args()

    model_path = args.model
    img_path = args.img

    inference_mnist(model_path, img_path)
