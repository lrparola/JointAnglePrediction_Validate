from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import configargparse


def parse_config(args=None):
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter

    cfg_parser = configargparse.YAMLConfigFileParser
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description='CMU-MBL Personalize Model',
                                      prog='JointAngle-Validation')

    parser.add_argument('--out_dir', type=str, default='output/optimization',
                        help='The optimizer used')
    parser.add_argument('-c', '--config',
                        required=True, is_config_file=True,
                        help='config file path')
    parser.add_argument('--use_cuda',
                        type=lambda arg: arg.lower() == 'true',
                        default=True,
                        help='Use CUDA for the computations')
    parser.add_argument('--optim_type', type=str, default='adam',
                        help='The optimizer used')
    parser.add_argument('--train_linear_layer', type=str, default=False,
                        help='train linear layer or all')
    parser.add_argument('--epochs', type=int, default=999,
                        help='The number of epochs to finetune')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='The size of mini batch')
    parser.add_argument('--lr', type=float, default=1.,
                        help='The learning rate for the algorithm')
    parser.add_argument('--beta', type=float, default=.9,
                        help='The momentum coefficient for the optimization')
    parser.add_argument('--activity', type=str, default='Walking', choices=['Walking', 'Running'],
                        help='The target activity')
    parser.add_argument('--joint', type=str, default='Knee', choices=['Ankle', 'Knee', 'Hip'],
                        help='The target joint')

    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict
