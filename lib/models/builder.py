from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from configs import constants as _C
from lib.models.pure_conv import CustomConv1D
from lib.models.pure_lstm import CustomLSTM

import sys
import os
import os.path as osp
import pickle

import torch


def build_nn_model(activity, joint,run_model=False,**kwargs):
    # Model checkpoint path
    model_dir = osp.join(_C.PATHS.MODEL_CHECKPOINT, activity)
    if run_model == False:
        
        state_dict_fname = osp.join(model_dir, f'{joint}_model.pt')
    if run_model == True:
        print('woo')
        state_dict_fname = 'model_dict_80_20.pt'
    kwargs_fname = osp.join(model_dir, f'{joint}_model_kwargs.pkl')
    norm_dict_fname = osp.join(model_dir, f'{joint}_norm_dict.pt')

    # Load kwargs
    with open(kwargs_fname, 'rb') as fopen:
        model_kwargs = pickle.load(fopen)

    # Build model
    net = globals()[model_kwargs['model_type']](**model_kwargs)

    # Load statedict
    net.load_state_dict(torch.load(state_dict_fname))

    # Load norm dict
    norm_dict = torch.load(norm_dict_fname)

    return net, norm_dict


def build_optimizer(net=None,
                    optim_type=None,
                    lr=None,
                    beta=None,
                    **kwargs):

    if optim_type == 'adam':
        if kwargs['train_linear_layer'] == True:
            try:
                optimizer = torch.optim.Adam(
                linear_out.parameters(), lr=lr, betas=(beta, 0.999))
            except:
                optimizer = torch.optim.Adam(
                lin_out.parameters(), lr=lr, betas=(beta, 0.999))
        else:
            optimizer = torch.optim.Adam(
            net.parameters(), lr=lr, betas=(beta, 0.999))
    else:
        raise NotImplementedError

    return optimizer
