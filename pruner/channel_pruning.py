import numpy as np
import argparse
import os

from nni.compression.pytorch.utils import count_flops_params
import torch.nn as nn
import torch

from src.net import LeNet


def print_info(model):
    dummy_input = torch.randn([1, 1, 28, 28])
    flops, params, _ = count_flops_params(model, dummy_input, verbose=True)
    print(f"Model FLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M")


def pruner(model, output_dir, ratio=0.5):
    """
    :param model:
    :param output_dir:
    :param ratio:
    :return: pruned model
    """
    # Step1, model pruning and get mask
    cfg = []
    cfg_mask = []
    input_mask = []
    layer_id = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):  # for conv
            out_channels = m.weight.data.shape[0]  # get the output
            weight_copy = m.weight.data.abs().clone().cpu().numpy()
            L1_norm = np.sum(weight_copy, axis=(1, 2, 3))  # compute the weight
            arg_max = np.argsort(L1_norm)  # sort by size
            nums_of_keep_channels = int((1 - ratio) * out_channels)  # get saved channels according to ratio
            cfg.append(nums_of_keep_channels)
            arg_max_rev = arg_max[::-1][:nums_of_keep_channels]
            mask = torch.zeros(out_channels)
            mask[arg_max_rev.tolist()] = 1
            cfg_mask.append(mask)
            layer_id += 1
        elif isinstance(m, nn.Linear):  # for linear pruning
            in_channels, out_channels = m.weight.data.shape[1], m.weight.data.shape[0]  # get input and output numbers

            # last linear
            if layer_id == 4:
                mask = torch.ones(out_channels)
                cfg.append(out_channels)
                cfg_mask.append(mask)
                continue

            """
            Generally, the number of output channels of the conv equal to the input channels of fc. 
            However, due to the special structure of LeNet, this situation is not satisfied.
            So, here process the input channels of fc separately.
            first fc layer, layer id is 2
            """
            if layer_id == 2:
                weight_copy = m.weight.data.abs().clone().cpu().numpy()
                L1_norm = np.sum(weight_copy, axis=0)  # for input numbers, so axis is 0
                arg_max = np.argsort(L1_norm)
                nums_of_keep_channels = int((1 - ratio) * in_channels)
                arg_max_rev = arg_max[::-1][:nums_of_keep_channels]
                mask = torch.zeros(in_channels)
                mask[arg_max_rev.tolist()] = 1
                input_mask.append(mask)  # make input mask for this special fc

            # process output numbers of fc layers
            weight_copy = m.weight.data.abs().clone().cpu().numpy()
            L1_norm = np.sum(weight_copy, axis=1)  # for input numbers, so axis is 1
            arg_max = np.argsort(L1_norm)
            nums_of_keep_channels = int((1 - ratio) * out_channels)
            cfg.append(nums_of_keep_channels)
            arg_max_rev = arg_max[::-1][:nums_of_keep_channels]
            mask = torch.zeros(out_channels)
            mask[arg_max_rev.tolist()] = 1
            cfg_mask.append(mask)
            layer_id += 1

    # Step2, build the new model
    new_model = LeNet(cfg=cfg)
    print("new_model:\n", new_model)

    # Step3, speed up the model, remove the 0 part of mask
    layer_id_in_cfg = 0
    start_mask = torch.ones(1)  # represents the input channel (single channel, so put into 1)
    end_mask = cfg_mask[layer_id_in_cfg]  # represents the output channel
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('Conv2d    In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))

            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))

            # get input channel from idx0, and then, get output channel from idx1
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            m1.bias.data = m0.bias.data[idx1.tolist()]

            layer_id_in_cfg += 1

            # update start mask and end mask
            start_mask = end_mask
            end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Linear):
            if layer_id_in_cfg == 2:
                start_mask = input_mask[0]
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('Linear    In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            w1 = m0.weight.data[:, idx0.tolist()].clone()
            w1 = w1[idx1.tolist(), :].clone()
            m1.weight.data = w1.clone()
            m1.bias.data = m0.bias.data[idx1.tolist()]

            layer_id_in_cfg += 1
            if layer_id_in_cfg == len(cfg):
                continue
            start_mask = end_mask
            end_mask = cfg_mask[layer_id_in_cfg]

    # save pruned model
    save_path = os.path.join(output_dir, 'pruned_model.ckpt')
    torch.save(new_model.state_dict(), save_path)
    print(f"\nsave pruned model to: {save_path}\n")

    return new_model


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', type=str, default='', help='checkpoints of model')
    parser.add_argument('--output-dir', type=str, default='model_data', help='checkpoints of pruned model')
    parser.add_argument('--ratio', type=float, default=0.5, help='pruning scale. (default: 0.5)')

    return parser


if __name__ == '__main__':
    # Create dir for saving model
    if not os.path.isdir('../model_data/'):
        os.makedirs('../model_data/')

    args = get_argparse().parse_args()

    # build model and load state dict if possible
    old_model = LeNet()
    if args.ckpt_path:
        state_dict = torch.load(args.ckpt_path)
        old_model.load_state_dict(state_dict)

    # prune
    new_model = pruner(old_model, args.output_dir, args.ratio)
    print(new_model)
