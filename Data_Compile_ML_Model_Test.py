# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 09:59:35 2022

@author: cmumbl
"""


set_current_path = 'C:\\Users\\lparola'
import torch 
import os
import pickle
import numpy as np
from time import time
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import pandas as pd
os.chdir(os.path.join(set_current_path,'Box','Lauren-Projects','Code','JointAnglePrediction_JOB'))

#from demo import butter_low
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#inputs 
args = {}

args['Activity'] = 'Walking'
args['Joint']='Knee'
args['Subject'] = 'PS002'
args['Side'] = 'Left'
SegJointDict = {'Hip': ['pelvis', 'thigh'], 'Knee': ['thigh', 'shank'], 'Ankle': ['shank', 'foot']}
#CNN

import pickle
from nn_models.models.pure_conv import CustomConv1D
from nn_models.models.pure_lstm import CustomLSTM
with open(os.path.join(os.getcwd(),'nn_models','models','checkpoints',args['Activity'],args['Joint']+'_model_kwargs.pkl'), 'rb') as file:
    kwargs = pickle.load(file)

#LSTM
# Load model
def butter_low(data, order=4, fc=5, fs=100):
    """
    Zero-lag butterworth filter for column data (i.e. padding occurs along axis 0).
    The defaults are set to be reasonable for standard optoelectronic data.
    """
        
    # Filter design
    b, a = butter(order, 2*fc/fs, 'low')
    # Make sure the padding is neither overkill nor larger than sequence length permits
    padlen = min(int(0.5*data.shape[0]), 200)
    print(padlen)
    # Zero-phase filtering with symmetric padding at beginning and end
    print(np.shape(data))
    filt_data = []

    filt_data =  filtfilt(b, a, data, padlen=padlen, axis=0)
    return filt_data

kwargs.update({'device': device})
LSTM_model = globals()[kwargs['model_type']](**kwargs)


LSTM_model.load_state_dict(torch.load(os.path.join(os.getcwd(),'nn_models','models','checkpoints',args['Activity'],args['Joint']+'_model.pt')))
LSTM_model.to('cuda')
subject_list =[i for i in os.listdir('F:\\ACLR_Pilot_1\\') if 'S00' in i]
for subject in subject_list:
    args['Subject'] =subject
#organize data aligned with mocap
    data_loc = 'F:\\ACLR_Pilot_1\\'+args['Subject']
    if 'P' in args['Subject']:
        data_loc = os.path.join(data_loc,'Three Month')
    data_path = os.path.join(data_loc,'Output_Data')
    result_dict = {}
    activity_dict = {}
    
    for activity in os.listdir(data_path):
        temp_path = os.path.join(data_path,activity)
        if 'Trial' in activity:
            activity_dict[activity] = {}
        
            for i in ['Left','Right']:
               args['Side'] = i
               acc1 = np.load(os.path.join(temp_path,'IMU',args['Side'].lower()+' lateral '+SegJointDict[args['Joint']][0],args['Side'].lower()+' lateral '+SegJointDict[args['Joint']][0]+' acc.npy'))
               gyr1 =  np.load(os.path.join(temp_path,'IMU',args['Side'].lower()+' lateral '+SegJointDict[args['Joint']][0],args['Side'].lower()+' lateral '+SegJointDict[args['Joint']][0]+' gyr.npy'))
               acc2 =np.load(os.path.join(temp_path,'IMU',args['Side'].lower()+' lateral '+SegJointDict[args['Joint']][1],args['Side'].lower()+' lateral '+SegJointDict[args['Joint']][1]+' acc.npy'))
               gyr2 = np.load(os.path.join(temp_path,'IMU',args['Side'].lower()+' lateral '+SegJointDict[args['Joint']][1],args['Side'].lower()+' lateral '+SegJointDict[args['Joint']][1]+' gyr.npy'))
               ground_truth = np.load(os.path.join(temp_path,'Mocap',args['Side'].lower()[0]+args['Joint'].lower(),args['Side'].lower()[0]+args['Joint'].lower()+' angle.npy'))
               inputs = []
               for data in [acc1, gyr1, acc2, gyr2]:
                   print((np.shape(data)))
                   mag = np.linalg.norm(data, axis=-1, keepdims=True)
                   _data = np.concatenate((data, mag), axis=-1)
                   print(np.shape(_data))
                   inputs += [_data]
        
               inputs = np.concatenate(inputs, axis=-1)
               inputs = torch.from_numpy(inputs).to(device=device).float()
        
        
        # Normalize input data
               norm_dict = torch.load(os.path.join(os.getcwd(),'nn_models','models','checkpoints',args['Activity'],args['Joint']+'_norm_dict.pt'))['params']
               print(str(norm_dict["x_mean"])+' +/-'+str(norm_dict['x_std']))
        
               inputs = (inputs - norm_dict['x_mean']) / norm_dict['x_std']
                   
               LSTM_model.eval()
               t1 = time()
               pred = LSTM_model(inputs)
               t2 = time()
        
        # Unnormalize prediction
               pred = pred * norm_dict['y_std'] + norm_dict['y_mean']
               pred = pred.detach().cpu().numpy()
               ground_truth = butter_low(ground_truth)
               pred = pd.DataFrame(data=pred.squeeze(0),columns=['Flexion','Add/Abd','Int/Ext Rot'])
               ground_truth = pd.DataFrame(data=ground_truth,columns=['Add/Abd','Int/Ext Rot','Flexion',])
               
               pred = pred - pred.mean(axis=0) + ground_truth.mean(axis=0)
               rmse = np.sqrt(np.square(pred - ground_truth).mean(axis=0))
               print(rmse)
               temp_dict = {}
               temp_dict['Prediction'] = pd.DataFrame(data=pred.squeeze(0),columns=['Flexion','Add/Abd','Int/Ext Rot'])
               temp_dict['Ground Truth'] = pd.DataFrame(data=ground_truth,columns=['Add/Abd','Int/Ext Rot','Flexion',])
               temp_dict['RMSE'] =rmse
               activity_dict[activity][args['Side']] = temp_dict
    
    
    for i in activity_dict:
        fig, ax = plt.subplots(3)
        plt.suptitle(args['Subject']+' '+i)
        ax[0].plot(activity_dict[i]['Left']['Prediction']['Flexion'],label='Left',color='red')
        ax[0].plot(activity_dict[i]['Right']['Prediction']['Flexion'],label='Right',color='red',linestyle=':')
        plt.legend()
        
        ax[0].plot(activity_dict[i]['Left']['Ground Truth']['Flexion'],color='blue')
        ax[0].plot(activity_dict[i]['Right']['Ground Truth']['Flexion'],color='blue',linestyle=':')
    
        ax[1].plot(activity_dict[i]['Left']['Prediction']['Add/Abd'],label='Left',color='red')
        ax[1].plot(activity_dict[i]['Right']['Prediction']['Add/Abd'],label='Right',color='red',linestyle=':')
        plt.legend()
        
        ax[1].plot(activity_dict[i]['Left']['Ground Truth']['Add/Abd'],color='blue')
        ax[1].plot(activity_dict[i]['Right']['Ground Truth']['Add/Abd'],color='blue',linestyle=':')
    
        ax[2].plot(activity_dict[i]['Left']['Prediction']['Int/Ext Rot'],label='Left',color='red')
        ax[2].plot(activity_dict[i]['Right']['Prediction']['Int/Ext Rot'],label='Right',color='red',linestyle=':')
        plt.legend()
        
        ax[2].plot(activity_dict[i]['Left']['Ground Truth']['Int/Ext Rot'],color='blue')
        ax[2].plot(activity_dict[i]['Right']['Ground Truth']['Int/Ext Rot'],color='blue',linestyle=':')
        
        plt.savefig('C:\\Users\\lparola\\Box\\Lauren-Projects\\Figures\\Trouble Shooting\\Ground Truth Noise'+subject+i+'.png')