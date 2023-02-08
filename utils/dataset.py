import torch
import torchvision
import torchvision.transforms as transforms

import cv2 as cv


def get_dataset_loader(batch_size):
    train_pipeline = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            transforms.Resize((32, 32))
        ])

    test_pipeline = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1325,), std=(0.3105,)),
            transforms.Resize((32, 32))
        ])

    train_set = torchvision.datasets.MNIST(root='./src/datasets', train=True, download=True, transform=train_pipeline)
    test_set = torchvision.datasets.MNIST(root='./src/datasets', train=False, download=True, transform=test_pipeline)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


if __name__ == '__main__':
    # VISUALIZATION
    train_loader, _ = get_dataset_loader(batch_size=1)
    img = train_loader.dataset.data[-1].numpy()
    print(img.shape)
    target = train_loader.dataset.targets[-1].item()
    resized_img = cv.resize(img, None, fx=10, fy=10)
    cv.imshow("img", resized_img)
    cv.waitKey()
    cv.destroyAllWindows()
