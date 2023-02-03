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
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from pdb import set_trace as st



def preprocess(subjects, is_train,group_label):
    buffer = 500
    if group_label == False:
        out_fname = _C.PATHS.TRAIN_DATA_LABEL if is_train else _C.PATHS.TEST_DATA_LABEL
    if group_label == 'ACL':
        out_fname = _C.PATHS.ACL_DATA_LABEL
        
    if group_label == 'HEALTHY':
        out_fname = _C.PATHS.HEALTHY_DATA_LABEL
    if group_label == 'ALL':
        out_fname = _C.PATHS.ALL_DATA_LABEL
    imus, angles, subj_act, seqs, seq_idx = [], [], [],[], 0

    for subject in subjects:
        subject_dir = glob.glob(osp.join(_C.PATHS.DATA, f'{subject}*'))[0]
        #_, trials, _ = next(os.walk(subject_dir))
        trials = [activity for activity in os.listdir(subject_dir) if 'Day' not in activity]
        for trial in trials:
            trial_dir = osp.join(subject_dir, trial)

            # Load IMU data
            temp_imus = []
            imu_dir = osp.join(subject_dir, trial, 'IMU')
            # fig,axs = plt.subplots(4)
            
            if subject == 'PS005':
                IMU_LIST = ['left posterior shank',
                'left posterior thigh',
                'right posterior shank',
                'right posterior thigh',]
                for val,sensor in enumerate(IMU_LIST):
                    acc = np.load(osp.join(imu_dir, sensor, f'{sensor} acc_trimmed.npy'))
                    gyr = np.load(osp.join(imu_dir, sensor, f'{sensor} gyr_trimmed.npy'))
                    imu = torch.from_numpy(np.concatenate((acc, gyr), axis=-1)).float()
                    temp_imus.append(imu[buffer:].unsqueeze(1))
            else:
                for val,sensor in enumerate(_C.DATA.IMU_LIST):
                    acc = np.load(osp.join(imu_dir, sensor, f'{sensor} acc_trimmed.npy'))
                    gyr = np.load(osp.join(imu_dir, sensor, f'{sensor} gyr_trimmed.npy'))
                    imu = torch.from_numpy(np.concatenate((acc, gyr), axis=-1)).float()
                    temp_imus.append(imu[buffer:].unsqueeze(1))

                # axs[val].plot(gyr[0:600,:],label=sensor+' gyr')
                # axs[val].set_ylabel(sensor+' gyr')
            # fig.savefig('imus.png')
            temp_imus = torch.cat(temp_imus, dim=1)
            imus.append(temp_imus)

            # Load Mocap angle data
            temp_angles = []
            mocap_dir = osp.join(subject_dir, trial, 'Mocap')
            # fig,axs = plt.subplots(2)
            for val,joint in enumerate(_C.DATA.JOINT_LIST):
                #MOCAP CS X-adduction/abduction Y- internal/external rotation Z-flexion/extension
                angle = np.load(osp.join(mocap_dir, joint, f'{joint} angle_trimmed.npy'))
                #change to IK code output set up outputted from from IK code
                angle[:,[0,1,2]] = angle[:,[2,0,1]]
                temp_angles.append(torch.from_numpy(angle[buffer:]).unsqueeze(1).float())
                # axs[val].plot(angle[0:600,:],label=joint)
                # axs[val].set_ylabel(joint)
            # fig.savefig('mocap.png')
            temp_angles = torch.cat(temp_angles, dim=1)
            angles.append(temp_angles)
            subj_act.append(subject+'_'+trial)

            # Put label to each trial
            seqs.append((torch.ones(temp_angles.shape[0]) * seq_idx).to(dtype=torch.int16))
            seq_idx += 1
    #import pdb; pdb.set_trace()        
    imu = torch.cat(imus, dim=0)
    angle = torch.cat(angles, dim=0)
    torch.save({
        'imu': imu,
        'angle': angle,
        'label':subj_act,
        'seq': torch.cat(seqs, dim=0)},
        out_fname
    )
def preprocess_train(subjects, is_train,group_label):
    buffer = 500
    if group_label == False:
        out_fname = _C.PATHS.TRAIN_DATA_LABEL if is_train else _C.PATHS.TEST_DATA_LABEL
    if group_label == 'ACL':
        out_fname = _C.PATHS.ACL_DATA_LABEL
        
    if group_label == 'HEALTHY':
        out_fname = _C.PATHS.HEALTHY_DATA_LABEL
    if group_label == 'ALL':
        out_fname = _C.PATHS.ALL_DATA_LABEL
    imus, angles, subj_act, seqs, seq_idx = [], [], [],[], 0

    for subject in subjects:
        subject_dir = glob.glob(osp.join(_C.PATHS.DATA, f'{subject}*'))[0]
        #_, trials, _ = next(os.walk(subject_dir))
        trials = [activity for activity in os.listdir(subject_dir) if 'Day' not in activity]
        for trial in trials:
            trial_dir = osp.join(subject_dir, trial)

            # Load IMU data
            temp_imus = []
            imu_dir = osp.join(subject_dir, trial, 'IMU')
            # fig,axs = plt.subplots(4)
            plt.figure()
            if subject == 'PS005':
                print('yeah')
                IMU_LIST = ['left posterior shank',
                'left posterior thigh',
                'right posterior shank',
                'right posterior thigh',]
                for val,sensor in enumerate(IMU_LIST):
                    acc = np.load(osp.join(imu_dir, sensor, f'{sensor} acc_trimmed.npy'))
                    gyr = np.load(osp.join(imu_dir, sensor, f'{sensor} gyr_trimmed.npy'))
                    if is_train:
                        start = 0
                        end = int(len(acc)*0.8)
                    else:
                        start= int(len(acc)*0.8)
                        end =-1
                    if 'left posterior shank' in sensor:
                        plt.plot(gyr[0:800,2],label='IMU')
                    imu = torch.from_numpy(np.concatenate((acc, gyr), axis=-1)).float()
                    temp_imus.append(imu[start:end].unsqueeze(1))
            else:
                for val,sensor in enumerate(_C.DATA.IMU_LIST):
                    acc = np.load(osp.join(imu_dir, sensor, f'{sensor} acc_trimmed.npy'))
                    gyr = np.load(osp.join(imu_dir, sensor, f'{sensor} gyr_trimmed.npy'))
                    if is_train:
                        start = 0
                        end = int(len(acc)*0.8)
                    else:
                        start= int(len(acc)*0.8)
                        end =-1
                    imu = torch.from_numpy(np.concatenate((acc, gyr), axis=-1)).float()

                    temp_imus.append(imu[start:end].unsqueeze(1))

                    if 'left lateral shank' in sensor:
                        plt.plot(gyr[0:800,2],label='IMU')
                # axs[val].plot(gyr[0:600,:],label=sensor+' gyr')
                # axs[val].set_ylabel(sensor+' gyr')
            # fig.savefig('imus.png')
            
            temp_imus = torch.cat(temp_imus, dim=1)
            imus.append(temp_imus)

            # Load Mocap angle data
            temp_angles = []
            mocap_dir = osp.join(subject_dir, trial, 'Mocap')
            # fig,axs = plt.subplots(2)
            for val,joint in enumerate(_C.DATA.JOINT_LIST):
                #MOCAP CS X-adduction/abduction Y- internal/external rotation Z-flexion/extension
                angle = np.load(osp.join(mocap_dir, joint, f'{joint} angle_trimmed.npy'))
                #change to IK code output set up outputted from from IK code
                angle[:,[0,1,2]] = angle[:,[2,0,1]]
                if is_train:
                        start = 0
                        end = int(len(angle)*0.8)
                        print(end)
                        #print('length 80%:'+str(end))
                else:
                        start= int(len(angle)*0.8)
                        end =-1
                        #print('length 20%:'+str(len(angle)-int(len(angle)*0.8)))
                #print(len(angle))
                temp_angles.append(torch.from_numpy(angle[start:end]).unsqueeze(1).float())
                
                if 'l' in joint:
                    plt.plot(angle[0:800,0],label='ground truth')
                plt.legend()
                plt.savefig(subject+trial+'.png')
                # axs[val].set_ylabel(joint)
            # fig.savefig('mocap.png')
            temp_angles = torch.cat(temp_angles, dim=1)
            angles.append(temp_angles)
            subj_act.append(subject+'_'+trial)

            # Put label to each trial
            seqs.append((torch.ones(temp_angles.shape[0]) * seq_idx).to(dtype=torch.int16))
            seq_idx += 1
            #import pdb; pdb.set_trace()
    #import pdb; pdb.set_trace()        
    imu = torch.cat(imus, dim=0)
    angle = torch.cat(angles, dim=0)
    torch.save({
        'imu': imu,
        'angle': angle,
        'label':subj_act,
        'seq': torch.cat(seqs, dim=0)},
        out_fname
    )  
    
def preprocess_two_overground(subjects, is_train,group_label):

    if group_label == False:
        out_fname = _C.PATHS.TRAIN_DATA_LABEL if is_train else _C.PATHS.TEST_DATA_LABEL
    if group_label == 'ACL':
        out_fname = _C.PATHS.ACL_DATA_LABEL
    if group_label == 'EXTRA_SUBJECTS':
        out_fname = _C.PATHS.EXTRA_SUBJECTS
    if group_label == 'HEALTHY':
        out_fname = _C.PATHS.HEALTHY_DATA_LABEL
    if group_label == 'ALL':
        out_fname = _C.PATHS.ALL_DATA_LABEL
    imus, subj_act, seqs, seq_idx = [], [],[], 0

    for subject in subjects:
        subject_dir = glob.glob(osp.join(_C.PATHS.DATA, f'{subject}*'))[0]
        trials = [day for day in os.listdir(subject_dir) if 'Overground' in day]
        for trial in trials:
            trial_dir = osp.join(subject_dir, trial)
            for bout in os.listdir(trial_dir):

                # Load IMU data
                temp_imus = []
                imu_dir = osp.join(subject_dir, trial, bout)
                # fig,axs = plt.subplots(4)

                for val,sensor in enumerate(_C.DATA.IMU_LIST):
                    acc = np.load(osp.join(imu_dir, sensor, f'{sensor} acc_trimmed.npy'))
                    gyr = np.load(osp.join(imu_dir, sensor, f'{sensor} gyr_trimmed.npy'))
                    start= int(len(acc)*0.8)
                    end =-1
                    imu = torch.from_numpy(np.concatenate((acc, gyr), axis=-1)).float()
                    temp_imus.append(imu[start:end].unsqueeze(1))

                    # axs[val].plot(gyr[0:600,:],label=sensor+' gyr')
                    # axs[val].set_ylabel(sensor+' gyr')
                # fig.savefig('imus.png')
                temp_imus = torch.cat(temp_imus, dim=1)
                imus.append(temp_imus)

                subj_act.append(subject+'_'+trial+'_'+bout)

                # Put label to each trial
                seqs.append((torch.ones(temp_imus.shape[0]) * seq_idx).to(dtype=torch.int16))
                seq_idx += 1
    #import pdb; pdb.set_trace()  

    imu = torch.cat(imus, dim=0)
    torch.save({
        'imu': imu,
        'label':subj_act,
        'seq': torch.cat(seqs, dim=0)},
        out_fname
    )

def preprocess_remote(subjects, is_train,group_label):

    if group_label == False:
        out_fname = _C.PATHS.TRAIN_DATA_LABEL if is_train else _C.PATHS.TEST_DATA_LABEL
    if group_label == 'ACL':
        out_fname = _C.PATHS.ACL_DATA_LABEL
        
    if group_label == 'HEALTHY':
        out_fname = _C.PATHS.HEALTHY_DATA_LABEL
    if group_label == 'ALL':
        out_fname = _C.PATHS.ALL_DATA_LABEL
    imus, subj_act, seqs, seq_idx = [], [],[], 0

    for subject in subjects:
        subject_dir = glob.glob(osp.join(_C.PATHS.DATA, f'{subject}*'))[0]
        trials = [day for day in os.listdir(subject_dir) if 'Day' in day]
        for trial in trials:
            trial_dir = osp.join(subject_dir, trial)
            for bout in os.listdir(trial_dir):

                # Load IMU data
                temp_imus = []
                imu_dir = osp.join(subject_dir, trial, bout)
                # fig,axs = plt.subplots(4)

                for val,sensor in enumerate(_C.DATA.IMU_LIST):
                    acc = np.load(osp.join(imu_dir, sensor, f'{sensor} acc_trimmed.npy'))
                    print(acc)
                    gyr = np.load(osp.join(imu_dir, sensor, f'{sensor} gyr_trimmed.npy'))
                    imu = torch.from_numpy(np.concatenate((acc, gyr), axis=-1)).float()
                    temp_imus.append(imu.unsqueeze(1))

                    # axs[val].plot(gyr[0:600,:],label=sensor+' gyr')
                    # axs[val].set_ylabel(sensor+' gyr')
                # fig.savefig('imus.png')
                temp_imus = torch.cat(temp_imus, dim=1)
                imus.append(temp_imus)

                subj_act.append(subject+'_'+trial+'_'+bout)

                # Put label to each trial
                seqs.append((torch.ones(temp_imus.shape[0]) * seq_idx).to(dtype=torch.int16))
                seq_idx += 1
    #import pdb; pdb.set_trace()  
    print(len(imu))
    imu = torch.cat(imus, dim=0)
    torch.save({
        'imu': imu,
        'label':subj_act,
        'seq': torch.cat(seqs, dim=0)},
        out_fname
    )

if __name__ == '__main__':
    train_subjects = ['PS001','PS005', 'PS004', 'PS002','HS001','HS005','HS006','HS007']
    test_subjects = ['PS006','HS003','HS004','PS003',]
    #preprocess(train_subjects, True,False)
    #preprocess(test_subjects, False,False)
    # preprocess_train(['PS001', 'PS003', 'PS004', 'PS002','PS005','HS001','HS005','HS006','HS007','PS006','HS003','HS004'], True,False)
    # preprocess_train(['PS001', 'PS003', 'PS004', 'PS002','PS005','HS001','HS005','HS006','HS007','PS006','HS003','HS004'], False,False)
    
    preprocess_two_overground(['PS001','HS003'], False,'EXTRA_SUBJECTS')
    # preprocess(['PS001', 'PS003', 'PS004', 'PS002','PS005','PS006'], False,'ACL')
    # preprocess(['HS001','HS005','HS006','HS007','HS003','HS004'], False,'HEALTHY')
    #preprocess_remote(['PS001', 'PS003', 'PS004', 'PS002','PS005','HS001','HS005','HS006','HS007','PS006','HS003','HS004'], False,'ALL')
    #preprocess_remote(['HS006'], False,'ALL')
