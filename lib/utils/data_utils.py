from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from configs import constants as _C

import torch

from pdb import set_trace as st

def normalize_angle(angle, norm_dict):
    angle -= norm_dict['params']['y_mean'].to(device=angle.device).squeeze(0)
    angle /= norm_dict['params']['y_std'].to(device=angle.device).squeeze(0)

    return angle
    
def normalize_new_data(imu_data1, imu_data2):
    acc1_mean = imu_data1[:, :3].mean(axis=0).append(np.mean(np.linalg.norm(imu_data1[:, :3])))
    gyr1_mean = imu_data1[:, 3:].mean(axis=0).append(np.mean(np.linalg.norm(imu_data1[:, 3:])))
    acc2_mean = imu_data2[:, :3].mean(axis=0).append(np.mean(np.linalg.norm(imu_data2[:, :3])))
    gyr2_mean = imu_data2[:, 3:].mean(axis=0).append(np.mean(np.linalg.norm(imu_data2[:, 3:])))
    
    norm_dict = {}
    norm_dict['params'] = {}
    norm_dict['params']['x_mean'] = [acc1_mean, gyr1_mean, acc2_mean, gyr2_mean]
    
    acc1_std = imu_data1[:, :3].std(axis=0).append(np.std(np.linalg.norm(imu_data1[:, :3])))
    gyr1_std = imu_data1[:, 3:].std(axis=0).append(np.std(np.linalg.norm(imu_data1[:, 3:])))
    acc2_std = imu_data2[:, :3].std(axis=0).append(np.std(np.linalg.norm(imu_data2[:, :3])))
    gyr2_std = imu_data2[:, 3:].std(axis=0).append(np.std(np.linalg.norm(imu_data2[:, 3:])))
    norm_dict['params']['x_std'] = [acc1_std, gyr1_std, acc2_std, gyr2_std]
    
    return norm_dict
    
    


def process_imu_data(imu_data1, imu_data2, norm_dict):
    concat_mag = lambda x: torch.cat((x, torch.norm(x, dim=-1).unsqueeze(-1)), dim=-1)
    acc1 = concat_mag(imu_data1[:, :3])
    gyr1 = concat_mag(imu_data1[:, 3:])
    acc2 = concat_mag(imu_data2[:, :3])
    gyr2 = concat_mag(imu_data2[:, 3:])

    imu = torch.cat((acc1, gyr1, acc2, gyr2), dim=-1)
    imu -= norm_dict['params']['x_mean'].to(device=imu.device).squeeze(0)
    imu /= norm_dict['params']['x_std'].to(device=imu.device).squeeze(0)

    return imu
