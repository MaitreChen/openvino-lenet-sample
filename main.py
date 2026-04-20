# IMPORT PACKAGES
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

import torch
import torch.nn as nn

from utils.dataloader import get_dataset_loader
from src.net import LeNet, MLP

from distiller.kd import distillation_loss, predict_teacher


def train(net, batch_size, num_epoch, learning_rate, optim, mode):
    device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
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

        # save best model
        save_path = ''
        if mode == 'train':
            save_path = 'model_data/best.ckpt'
        elif mode == 'retrain':
            save_path = 'model_data/best_pruned.ckpt'
        elif mode == 'mlp':
            save_path = 'model_data/best_mlp.ckpt'

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), save_path)

            print("-" * 10)
            print(f"The best accuracy is: {100. * best_acc:.2f} %")
            print(f"save best model to {save_path}\n")
            print("-" * 10)

    print(f'Finished {mode}ing!')

    return train_acc_set, train_loss_set, test_acc_set


def train_kd(student, teacher, batch_size, num_epoch, learning_rate, optim, temp, alpha):
    device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    student.to(device)
    teacher.to(device)
    teacher.eval()
    print(f"KD training on device: {device}")

    if optim == "sgd":
        optimizer = torch.optim.SGD(student.parameters(), lr=learning_rate, momentum=0.9)
    elif optim == "adam":
        optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    else:
        raise ValueError("optim-policy must be in [sgd | adam]")

    test_acc = 0.0
    best_acc = 0.0

    for epoch in range(num_epoch):
        total_loss = 0.0
        train_acc = 0.0

        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        student.train()
        for i, (x, y) in loop:
            inputs, targets = x.to(device), y.to(device)
            outputs = student(inputs)

            with torch.no_grad():
                teacher_outputs = predict_teacher(teacher, inputs).detach()

            loss = distillation_loss(outputs, targets, teacher_outputs, temp=temp, alpha=alpha)

            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

            _, prediction = outputs.max(1)
            num_correct = (prediction == targets).sum().item()
            acc = num_correct / batch_size
            train_acc += acc

            loop.set_description(f"KD Epoch [{epoch + 1}/{num_epoch}]")
            loop.set_postfix(Loss=total_loss / (i + batch_size))

        # test
        test_acc = test(student, test_loader, device)

        # save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = 'model_data/best_kd_mlp.ckpt'
            torch.save(student.state_dict(), save_path)

            print("-" * 10)
            print(f"The best KD accuracy is: {100. * best_acc:.2f} %")
            print(f"save best KD student to {save_path}\n")
            print("-" * 10)

    print('Finished KD training!')


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


def get_argparse():
    parser = argparse.ArgumentParser()

    # train options
    parser.add_argument('--batch-size', default=256, type=int, help='batch size for training')
    parser.add_argument('--epoch', default=1, type=int, help='number of epochs for training')
    parser.add_argument('--optim-policy', type=str, default='sgd', help='optimizer for training. [sgd | adam]')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--use-gpu', action='store_true', default=True, help='turn on flag to use GPU')

    # prune options
    parser.add_argument('--prune', action='store_true', default=False, help='turn on flag to prune')
    parser.add_argument('--output-dir', type=str, default='model_data', help='checkpoints of pruned model')
    parser.add_argument('--ratio', type=float, default=0.5, help='pruning scale. (default: 0.5)')
    parser.add_argument('--retrain-mode', type=int, default=0, help='[train from scratch:0 | fine-tune:1]')
    parser.add_argument('--p-epoch', default=2, type=int, help='number of epochs for retraining')
    parser.add_argument('--p-lr', default=0.01, type=float, help='learning rate for retraining')

    # kd options
    parser.add_argument('--kd', action='store_true', default=False, help='turn on flag to use knowledge distillation')
    parser.add_argument('--teacher-ckpt', type=str, default='model_data/best.ckpt', help='teacher ckpt path')
    parser.add_argument('--kd-epoch', default=2, type=int, help='number of epochs for KD training')
    parser.add_argument('--kd-lr', default=0.01, type=float, help='learning rate for KD training')
    parser.add_argument('--temp', default=5.0, type=float, help='distillation temperature')
    parser.add_argument('--alpha', default=0.7, type=float, help='distillation alpha')

    # mlp options (no KD)
    parser.add_argument('--mlp', action='store_true', default=False, help='turn on flag to train MLP without KD')
    parser.add_argument('--mlp-epoch', default=2, type=int, help='number of epochs for MLP training')
    parser.add_argument('--mlp-lr', default=0.01, type=float, help='learning rate for MLP training')

    return parser


if __name__ == "__main__":
    # Get arguments
    args = get_argparse().parse_args()

    # Create dir for saving model
    if not os.path.isdir('model_data/'):
        os.makedirs('model_data/')

    # Load dataset
    train_loader, test_loader = get_dataset_loader(batch_size=args.batch_size)

    # Build model
    net = LeNet()

    # Start training
    train_acc_set, train_loss_set, test_acc_set = train(net, args.batch_size, args.epoch, args.lr, args.optim_policy,
                                                        'train')

    # kd: LeNet teacher -> MLP student
    if args.kd:
        teacher = LeNet()
        teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location='cpu'))
        student = MLP()
        train_kd(student, teacher, args.batch_size, args.kd_epoch, args.kd_lr, args.optim_policy, args.temp, args.alpha)

    # mlp: train baseline without KD
    if args.mlp:
        mlp = MLP()
        train(mlp, args.batch_size, args.mlp_epoch, args.mlp_lr, args.optim_policy, 'mlp')

    # prune and retrain to restore accuracy
    if args.prune:
        from pruner.channel_pruning import pruner

        net.load_state_dict(torch.load('model_data/best.ckpt'))
        pruned_model = pruner(net, args.output_dir, args.ratio)

        if args.retrain_mode == 1:
            pruned_state_dict = torch.load(os.path.join(args.output_dir, 'pruned_model.ckpt'))
            pruned_model.load_state_dict(pruned_state_dict)

        # retrain
        train_acc_set, train_loss_set, test_acc_set = train(pruned_model, args.batch_size, args.p_epoch, args.p_lr,
                                                            args.optim_policy, 'retrain')
