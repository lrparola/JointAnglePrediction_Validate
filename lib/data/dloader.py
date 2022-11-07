from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from configs import constants as _C
from lib.utils.data_utils import normalize_angle, process_imu_data

import torch
import numpy as np
from skimage.util.shape import view_as_windows

from pdb import set_trace as st


class Dataset(torch.utils.data.Dataset):
    def __init__(self, label_pth, joint, norm_dict, **kwargs):
        super(Dataset, self).__init__()

        self.labels = torch.load(label_pth)
        self.joint = joint
        self.norm_dict = norm_dict
        self.prepare_sequence_batch(kwargs.get('input_length', 400))


    def prepare_sequence_batch(self, seq_length):
        self.seq_indices = []
        seqs = self.labels['seq']

        seqs_unique, group = np.unique(
            seqs, return_index=True)
        perm = np.argsort(group)
        group_perm = group[perm]
        indices = np.split(
            np.arange(0, seqs.shape[0]), group_perm[1:]
        )
        for idx in range(len(seqs_unique)):
            indexes = indices[idx]
            if indexes.shape[0] < seq_length: continue
            chunks = view_as_windows(
                indexes, (seq_length), step=seq_length // 4
            )
            start_finish = chunks[:, (0, -1)].tolist()
            self.seq_indices += start_finish


    def __len__(self):
        return len(self.seq_indices)

    def __getitem__(self, index):
        return self.get_single_sequence(index)


    def get_single_sequence(self, index):
        start_index, end_index = self.seq_indices[index]
        all_imu = self.labels['imu'][start_index:end_index+1]
        all_angle = self.labels['angle'][start_index:end_index+1]

        imus, angles = [], []
        for side in ['left', 'right']:
            trg_joint = side[0] + self.joint.lower()
            temp_imu = []
            for sensor in _C.DATA.JOINT_IMU_MAPPER[trg_joint]:
                temp_imu.append(all_imu[:, _C.DATA.IMU_LIST.index(sensor)])
            imu = process_imu_data(*temp_imu, self.norm_dict)
            imus.append(imu)

            angle = all_angle[:, _C.DATA.JOINT_LIST.index(trg_joint)]
            angle = normalize_angle(angle, self.norm_dict)
            angles.append(angle)

        imus = torch.stack(imus)
        angles = torch.stack(angles)

        return imus, angles


def setup_validation_data(norm_dict=None,
                          joint=None,
                          batch_size=None,
                          **kwargs):
    n_workers = 0

    train_dataset = Dataset(_C.PATHS.TRAIN_DATA_LABEL, joint, norm_dict, **kwargs)
    train_dloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size // 2,
        num_workers=n_workers,
        shuffle=True,
        pin_memory=True,
    )

    return train_dloader
