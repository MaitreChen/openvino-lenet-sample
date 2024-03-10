from time import time
import cv2 as cv
import torch
import argparse

from src.net import Net


def get_test_transform():
    from torchvision import transforms
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Resize((28, 28))
                               ])


def inference_mnist(model_path, img_path, device):
    # load model and params
    net = Net()
    net.to(device)
    net.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    net.eval()

    # load image
    img = cv.imread(img_path, 0)
    src = img.copy()
    img = get_test_transform()(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    # print(img.shape)

    # Inference
    start_time = time()
    outs = net(img)
    end_time = time()
    print(f"Inference time: {(end_time - start_time) * 1000:.6f} ms")
    print(f"The prediction digit: {torch.argmax(outs)}")

    res_img = cv.resize(src, None, fx=10, fy=10)
    rgb_img = cv.cvtColor(res_img, cv.COLOR_GRAY2BGR)
    cv.putText(rgb_img, "Prediction:" + str(torch.argmax(outs).item()), (0, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6,
               (0, 0, 255), 2)
    cv.imshow("Result", rgb_img)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True, help="ckpt、pth、pt、pkl")
    parser.add_argument('-i', '--img', type=str, required=True, help="test image")
    parser.add_argument('-d', '--device', type=str, required=True, help="cpu or cuda")
    args = parser.parse_args()

    model_path = args.model
    img_path = args.img
    device = args.device

    inference_mnist(model_path, img_path, device)
