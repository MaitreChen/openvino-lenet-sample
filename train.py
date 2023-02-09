# IMPORT PACKAGES
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from src.net import Net
from utils.dataset import get_dataset_loader

import argparse
import os


# train
def trainer(net, batch_size, num_epoch, device, learning_rate, optim='sgd'):
    # 训练设备
    net.to(device)
    print(f"Training on device: {device}")

    # 优化器
    if optim == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    elif optim == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    train_acc_set = []
    test_acc_set = []
    train_loss_set = []

    # 训练
    test_acc = 0.0
    best_acc = 0.0
    for epoch in range(num_epoch):
        total_loss = 0.0
        train_acc = 0.0

        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        net.train()
        for i, (x, y) in loop:
            inputs, targets = x.to(device), y.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

            _, prediction = outputs.max(1)
            num_correct = (prediction == targets).sum().item()
            acc = num_correct / batch_size
            train_acc += acc

            loop.set_description(f"Epoch [{epoch + 1}/{num_epoch}]")
            loop.set_postfix(Loss=total_loss / (i + batch_size))

        train_acc_set.append(train_acc / len(train_loader))
        train_loss_set.append(loss.item())

        # 测试
        test_acc = test(net, test_loader, device)
        test_acc_set.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc

            torch.save(net.state_dict(), 'model/best.ckpt')

    print('Finished Training\n')
    print('Save best model to model file!')
    print(f"The best accuracy is: {100. * best_acc:.2f} %")

    return train_acc_set, train_loss_set, test_acc_set


def test(net, data_loader, device):
    net.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            _, prediction = torch.max(outputs, 1)

            correct += (prediction == labels).sum().item()
            total += labels.size(0)

    return correct / total


def plot_loss_acc():
    epoch_set = list(range(1, len(train_acc_set) + 1))
    plt.plot(epoch_set, train_acc_set, lw=1.5, c='r', label='train-acc')
    plt.plot(epoch_set, train_loss_set, lw=1.5, c='g', label='train-loss')
    plt.plot(epoch_set, test_acc_set, lw=1.5, c='b', label='test-acc')
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim((1, len(train_acc_set)))
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    if not os.path.isdir('model/'):
        os.makedirs('model/')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=64, type=int, required=False)
    parser.add_argument('--epoch', default=20, type=int, required=False)
    parser.add_argument('--lr', default=0.01, type=float, required=False)
    args = parser.parse_args()

    batch_size = args.batch
    epochs = args.epoch
    learning_rate = args.lr
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()

    # Initialize network
    net = Net()

    # Load dataset
    train_loader, test_loader = get_dataset_loader(batch_size=batch_size)

    train_acc_set, train_loss_set, test_acc_set = trainer(net, batch_size, epochs, device, learning_rate)

    plot_loss_acc()
