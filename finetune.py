from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from configs.cmd_parse import parse_config
from lib.data.dloader import setup_validation_data
from lib.models.builder import build_nn_model, build_optimizer

import sys
import os
import os.path as osp

import torch
import torch.nn as nn
from pdb import set_trace as st


def train_one_epoch(net, optimizer, train_dloader, device, **kwargs):
    net.train()
    criterion = nn.CrossEntropyLoss()
    for _iter, batch in enumerate(train_dloader):
        x_in, y_gt = batch
        import pdb; pdb.set_trace()
        # Concatenate left and right leg
        x_in = torch.reshape(-1, *x_in.shape[-2:]).to(device)
        y_gr = torch.reshape(-1, *y_gt.shape[-2:]).to(device)

        # Predict
        y_pred = net(x_in)

        # Calculate loss
        loss = criterion(y_pred, y_gr)
       


def main(**kwargs):
    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    device = 'cuda' if use_cuda else 'cpu'
    net, norm_dict = build_nn_model(**kwargs)
    net.to(device=device)

    optimizer = build_optimizer(net, **kwargs)

    train_dloader = setup_validation_data(norm_dict, **kwargs)
    # eval_dloader

    for _ in range(1, kwargs.get('epochs') + 1):
        train_one_epoch(net, optimizer, train_dloader, device, **kwargs)


if __name__ == '__main__':
    args = parse_config()
    main(**args)
