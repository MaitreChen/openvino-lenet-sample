# IMPORT PACKAGES
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

import torch
import torch.nn as nn

from utils.dataset import get_dataset_loader
from src.net import LeNet


def trainer(net, batch_size, num_epoch, learning_rate, optim='sgd'):
    device = 'cuda' if args.use_gpu else 'cpu'
    net.to(device)
    print(f"Training on device: {device}")

    # define loss function
    criterion = nn.CrossEntropyLoss()

    if optim == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    elif optim == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    train_acc_set = []
    test_acc_set = []
    train_loss_set = []

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

        # test
        test_acc = test(net, test_loader, device)
        test_acc_set.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc

            save_path = 'model_data/best.ckpt'
            torch.save(net.state_dict(), save_path)
            print(f"The best accuracy is: {100. * best_acc:.2f} %")

    print(f"save best model to model_data!")
    print('Finished Training\n')

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
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()


def get_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', default=64, type=int, help='batch size for training')
    parser.add_argument('--epoch', default=4, type=int, help='number of epochs for training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='turn on flag to use GPU')
    parser.add_argument('--prune', action='store_true', default=False, help='turn on flag to prune')

    return parser


if __name__ == "__main__":
    # Create dir for saving model
    if not os.path.isdir('model_data/'):
        os.makedirs('model_data/')

    # Get arguments
    args = get_argparse().parse_args()

    # Build model
    net = LeNet()

    # Load dataset
    train_loader, test_loader = get_dataset_loader(batch_size=args.batch_size)

    # Start training
    train_acc_set, train_loss_set, test_acc_set = trainer(net, args.batch_size, args.epoch, args.lr)

    # prune and retrain to restore accuracy
    # if args.prune:

    # Plot result
    plot_loss_acc()
