from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from configs.cmd_parse import parse_config
from lib.data.dloader import setup_validation_data,setup_confirmation_data, setup_remote_data,setup_overground_data
from lib.models.builder import build_nn_model, build_optimizer
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import torch
import numpy
from pdb import set_trace as st
from scipy.signal import find_peaks
def plot_average_gait_cycle(y_pred,y_gt,population,activity_label):
    fig, ax = plt.subplots()
    for i in range(2):
        yswing_peak,_ = find_peaks(y_pred[i,:],height=50,distance=60)
        
        ystance_peak,_ = find_peaks(-y_pred[i,:],height=-20,distance=30)
        
        
        ystance_fix = [ystance_peak[ystance_peak > val][0] for val in yswing_peak if any(ystance_peak > val)]
        
        ystance_peak = ystance_fix
        
        
        
        gtswing_peak,_ = find_peaks(y_gt[i,:],height=40,distance=60)

        gtstance_peak,_ = find_peaks(-y_gt[i,:],height=-20,distance=30)

        gtstance_fix = [gtstance_peak[gtstance_peak > val][0] for val in gtswing_peak if any(gtstance_peak > val)]
        gtstance_peak=gtstance_fix

        step_cycle_matrix_y = []
        peak_height_y = []
        #print(len(y_pred[i,:]))
        plt.figure()
        plt.plot(y_pred[i,:])
        plt.savefig('test.png')
        for val in range(1,len(ystance_peak)):

            temp_range = y_pred[i,:][ystance_peak[val-1]:ystance_peak[val]]
            _, height = find_peaks(temp_range,height=[0,50], distance =60)
            peak_height_y = numpy.append(peak_height_y,height['peak_heights'])
            x = numpy.linspace(0,100,len(temp_range))
            f = interp1d(x, temp_range, kind='cubic')
            xnew = numpy.linspace(0,100,100)
            data = f(xnew)
            if val == 1 :
                    step_cycle_matrix_y = [data]
            else:
                    step_cycle_matrix_y = numpy.concatenate((step_cycle_matrix_y, [data]),axis=0)
        print(step_cycle_matrix_y)
        step_cycle_matrix_gt = []
        peak_height_gt = []
        #print(len(gtstance_peak))
        for val in range(1,len(gtstance_peak)):
                temp_range =y_gt[i,:][gtstance_peak[val-1]:gtstance_peak[val]]
                _, height = find_peaks(temp_range,height=[0,50], distance =60)
                peak_height_gt = numpy.append(peak_height_gt,height['peak_heights'])
                x = numpy.linspace(0,100,len(temp_range))
                f = interp1d(x, temp_range, kind='cubic')
                xnew = numpy.linspace(0,100,100)
                data = f(xnew)
                if val == 1 :
                        step_cycle_matrix_gt = [data]
                else:
                        step_cycle_matrix_gt = numpy.concatenate((step_cycle_matrix_gt, [data]),axis=0)
        step_cycle_matrix_y=numpy.array(step_cycle_matrix_y)
        step_cycle_matrix_gt=numpy.array(step_cycle_matrix_gt)
        #print(numpy.shape(step_cycle_matrix_gt))
        plt.figure() 
        # fig.suptitle(subject)
        
        ax.fill_between(numpy.linspace(0,100,100),numpy.mean(step_cycle_matrix_y,axis=0)-numpy.std(step_cycle_matrix_y,axis=0), numpy.mean(step_cycle_matrix_y,axis=0)+numpy.std(step_cycle_matrix_y,axis=0), alpha=0.2,color='blue')
        ax.plot(numpy.linspace(0,100,100),numpy.mean(step_cycle_matrix_y,axis=0),c='b',label = 'IMU Estimate')
        # plt.plot(numpy.linspace(0,100,100),numpy.mean(step_cycle_matrix_y,axis=0)-numpy.std(step_cycle_matrix_y,axis=0),'b--',label = '+/- Standard Deviation')
        # plt.plot(numpy.linspace(0,100,100),numpy.mean(step_cycle_matrix_y,axis=0)+numpy.std(step_cycle_matrix_y,axis=0),'b--')
       
        ax.fill_between(numpy.linspace(0,100,100),numpy.mean(step_cycle_matrix_gt,axis=0)-numpy.std(step_cycle_matrix_gt,axis=0), numpy.mean(step_cycle_matrix_gt,axis=0)+numpy.std(step_cycle_matrix_gt,axis=0), alpha=0.2,color='green')
        ax.plot(numpy.linspace(0,100,100),numpy.mean(step_cycle_matrix_gt,axis=0),color='green',label = 'Ground Truth')
        
        # plt.plot(numpy.linspace(0,100,100),numpy.mean(step_cycle_matrix_gt,axis=0),c='g',label = 'mean')
        # plt.plot(numpy.linspace(0,100,100),numpy.mean(step_cycle_matrix_gt,axis=0)-numpy.std(step_cycle_matrix_gt,axis=0),'g--',label = '+/- Standard Deviation')
        # plt.plot(numpy.linspace(0,100,100),numpy.mean(step_cycle_matrix_gt,axis=0)+numpy.std(step_cycle_matrix_gt,axis=0),'g--')
        plt.legend()
        fig.savefig(activity_label+'.png')
    # axs[1].set_ylabel('Left '+str(round(np.mean(np.array(peak_height_l))))+' degrees')
    return
    
    
def export(y_pred,file_name,y_gr=False,is_remote=False):

        if 'PS003' in file_name or 'PS005' in file_name or 'PS006' in file_name or 'HS006' in file_name:
            print(file_name)
            dom = 0
            rec = 1
        else:
            dom=1
            rec = 0
        file_name.split('_')[0]

        if os.path.isdir(os.path.join('C:\\Users\\lparola\\JointAnglePrediction_Validate\\exported',file_name.split('_')[0])) == False:
            os.mkdir(os.path.join('C:\\Users\\lparola\\JointAnglePrediction_Validate\\exported',file_name.split('_')[0]))
        path = os.path.join('C:\\Users\\lparola\\JointAnglePrediction_Validate\\exported',file_name.split('_')[0])
        if is_remote:
            if os.path.isdir(os.path.join(path,'Remote')) == False:
                os.mkdir(os.path.join(path,'Remote'))
            path = os.path.join(path,'Remote')
            if os.path.isdir(os.path.join(path,file_name.split('_')[1])) == False:
                os.mkdir(os.path.join(path,file_name.split('_')[1]))
            if os.path.isdir(os.path.join(path,file_name.split('_')[1],file_name.split('_')[2])) == False:
                os.mkdir(os.path.join(path,file_name.split('_')[1],file_name.split('_')[2]))
            path = os.path.join(path,file_name.split('_')[1],file_name.split('_')[2])
            numpy.save(os.path.join(path,file_name+' dominant prediction.npy'),y_pred[dom,:])
            numpy.save(os.path.join(path,file_name+' nondominant prediction.npy'),y_pred[rec,:])
        else: 
            if os.path.isdir(os.path.join(path,file_name.split('_')[1])) == False:
                os.mkdir(os.path.join(path,file_name.split('_')[1]))
            path = os.path.join(path,file_name.split('_')[1])

            numpy.save(os.path.join(path,file_name+' dominant prediction.npy'),y_pred[dom,:])
            numpy.save(os.path.join(path,file_name+' dominant ground truth.npy'),y_gr[dom,:])
            numpy.save(os.path.join(path,file_name+' nondominant prediction.npy'),y_pred[rec,:])
            numpy.save(os.path.join(path,file_name+' nondominant ground truth.npy'),y_gr[rec,:])
        print('Exported for '+file_name)
        return
def test_model(net, test_dloader, device,population):
    net.eval()
    criterion = nn.MSELoss()
    rmses = []
    
    with torch.no_grad():
        for val, batch in enumerate(test_dloader):
            x_in, y_gt, activity_label = batch
            # print(activity_label)
            
        
        # Concatenate left and right leg
            x_in = x_in.reshape(-1, *x_in.shape[-2:]).to(device)
            y_gr = y_gt.reshape(-1, *y_gt.shape[-2:]).to(device)
            y_pred = net.forward(x_in)
                        
            y_pred = test_dloader.dataset.normalizer.unnormalize_output(y_pred.detach().cpu())
            y_gr = test_dloader.dataset.normalizer.unnormalize_output(y_gr.detach().cpu())

            rmse = ((y_pred - y_gr) ** 2).mean(1) ** 0.5
   
            rmses.append(rmse)

                # plt.figure()
                # plt.plot(y_pred[1,:800,0].numpy())
                # plt.plot(y_gr[1,:800,0].numpy())
                # plt.plot(y_pred[0,:800,0].numpy())
                # plt.plot(y_gr[0,:800,0].numpy())
                # plt.ylabel('Angle (deg)')
                
                # plt.savefig('full'+activity_label[0]+'.png')
            #plot_average_gait_cycle(y_pred[:,:,0].numpy(),y_gr[:,:,0].numpy(),population+str(val),activity_label[0])
            export(y_pred[:,:,0],activity_label[0],y_gr[:,:,0],is_remote=False)
        #import pdb; pdb.set_trace()
    rmses = torch.cat(rmses)
    print(rmses[:,0])

    rmse = ((y_pred - y_gr) ** 2).mean(1) ** 0.5

    return rmses[:,0].mean().item()
    
def apply_model(net, test_dloader, device,population):
    net.eval()

    criterion = nn.MSELoss()
    rmses = []
    
    with torch.no_grad():

        for val, batch in enumerate(test_dloader):

            x_in, activity_label = batch

            
        
        # Concatenate left and right leg
            x_in = x_in.reshape(-1, *x_in.shape[-2:]).to(device)
           
            y_pred = net.forward(x_in)
                        
            y_pred = test_dloader.dataset.normalizer.unnormalize_output(y_pred.detach().cpu())
           

            # plt.figure()
            # plt.plot(y_pred[1,:800,0].numpy())

            # plt.plot(y_pred[0,:800,0].numpy())
            # plt.ylabel('Angle (deg)')
            # plt.savefig('full'+activity_label[0]+'.png')
            #import pdb; pdb.set_trace()
                #plot_average_gait_cycle(y_pred[:,:,0].numpy(),y_gr[:,:,0].numpy(),population+str(val),activity_label[0])
            export(y_pred[:,:,0],activity_label[0],False,True)




    return 

def main(**kwargs):
    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)
    device = 'cuda' if use_cuda else 'cpu'

    net, norm_dict = build_nn_model(run_model=True,**kwargs)
    net.to(device=device)

    optimizer = build_optimizer(net, **kwargs)
    #run this to load data with a ground truth mocap
    #all_dloader = setup_confirmation_data(norm_dict, device=device, **kwargs)
    
    
    
    #run this to load data from in lab data without ground truth (still splits 80/20)
    #all_dloader = setup_overground_data(norm_dict, device=device, **kwargs)
    
    
    
    #run this to load data  without a ground truth
    # all_dloader = setup_remote_data(norm_dict,device=device,
                           # **kwargs)



    #run test model to apply model to data that also has ground truth
    #print(f'All RMSE: {test_model(net,all_dloader,device,False): .2f} (deg)')
    
    
    #run this function to only apply model, not compare RMSE with ground truth
    apply_model(net,all_dloader,device,False)
    return

       


if __name__ == '__main__':
    args = parse_config()
    main(**args)