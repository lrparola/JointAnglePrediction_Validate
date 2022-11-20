from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from configs import constants as _C

import glob
import sys
import os
import os.path as osp

import torch
import numpy as np

from pdb import set_trace as st


def load_imu(data_dir):
    pass


def preprocess(subjects, is_train):
    out_fname = _C.PATHS.TRAIN_DATA_LABEL if is_train else _C.PATHS.TEST_DATA_LABEL
    imus, angles, seqs, seq_idx = [], [], [], 0

    for subject in subjects:
        subject_dir = glob.glob(osp.join(_C.PATHS.DATA, f'{subject}*'))[0]
        _, trials, _ = next(os.walk(subject_dir))
        for trial in trials:
            trial_dir = osp.join(subject_dir, trial)

            # Load IMU data
            temp_imus = []
            imu_dir = osp.join(subject_dir, trial, 'IMU')
            for sensor in _C.DATA.IMU_LIST:
                acc = np.load(osp.join(imu_dir, sensor, f'{sensor} acc.npy'))
                gyr = np.load(osp.join(imu_dir, sensor, f'{sensor} gyr.npy'))
                imu = torch.from_numpy(np.concatenate((acc, gyr), axis=-1)).float()
                temp_imus.append(imu.unsqueeze(1))
            temp_imus = torch.cat(temp_imus, dim=1)
            imus.append(temp_imus)

            # Load Mocap angle data
            temp_angles = []
            mocap_dir = osp.join(subject_dir, trial, 'Mocap')
            for joint in _C.DATA.JOINT_LIST:
                #MOCAP CS X-adduction/abduction Y- internal/external rotation Z-flexion/extension
                angle = np.load(osp.join(mocap_dir, joint, f'{joint} angle.npy'))
                #change to IK code output set up outputted from from IK code
                angle[:,[0,1,2]] = angle[:,[2,0,1]]
                temp_angles.append(torch.from_numpy(angle).unsqueeze(1).float())
            temp_angles = torch.cat(temp_angles, dim=1)
            angles.append(temp_angles)

            # Put label to each trial
            seqs.append((torch.ones(temp_angles.shape[0]) * seq_idx).to(dtype=torch.int16))
            seq_idx += 1

    torch.save({
        'imu': torch.cat(imus, dim=0),
        'angle': torch.cat(angles, dim=0),
        'seq': torch.cat(seqs, dim=0)},
        out_fname
    )

if __name__ == '__main__':
    train_subjects = ['HS001', 'HS003', 'HS004', 'HS005']
    test_subjects = ['HS006', 'HS007']
    preprocess(train_subjects, True)
    preprocess(test_subjects, False)
