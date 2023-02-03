from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from configs.cmd_parse import parse_config
from lib.data.dloader import setup_validation_data,setup_confirmation_data
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
def plot_average_gait_cycle(y_pred,y_gt,population):
    print(numpy.shape(y_gt))

    yswing_peak,_ = find_peaks(y_pred,height=50,distance=60)
    
    ystance_peak,_ = find_peaks(-y_pred,height=-20,distance=30)
    
    
    ystance_fix = [ystance_peak[ystance_peak > val][0] for val in yswing_peak if any(ystance_peak > val)]
    
    ystance_peak = ystance_fix
    
    
    
    

    # print(numpy.shape(y_gt))
    plt.figure()
    plt.plot(y_pred)
    plt.savefig('test_pred.png')
    
    gtswing_peak,_ = find_peaks(y_gt,height=40,distance=60)

    gtstance_peak,_ = find_peaks(-y_gt,height=-20,distance=30)

    gtstance_fix = [gtstance_peak[gtstance_peak > val][0] for val in gtswing_peak if any(gtstance_peak > val)]
    gtstance_peak=gtstance_fix
    print(numpy.shape(gtstance_peak))
    # plt.figure()
    # plt.plot(trial_dict['lknee'])
    # plt.scatter(lstance_fix,trial_dict['lknee'][lstance_fix])
    # plt.scatter(lswing_peak,trial_dict['lknee'][lswing_peak])
    # plt.plot(trial_dict['rknee'])
    # plt.scatter(rswing_peak,trial_dict['rknee'][rswing_peak])
    # plt.scatter(rstance_fix,trial_dict['rknee'][rstance_fix])
    # plt.title(subject+' '+trial)
    step_cycle_matrix_y = []
    peak_height_y = []
    for val in range(1,len(ystance_peak)):

        temp_range = y_pred[ystance_peak[val-1]:ystance_peak[val]]
        _, height = find_peaks(temp_range,height=[0,40], distance =60)
        peak_height_y = numpy.append(peak_height_y,height['peak_heights'])
        x = numpy.linspace(0,100,len(temp_range))
        f = interp1d(x, temp_range, kind='cubic')
        xnew = numpy.linspace(0,100,100)
        data = f(xnew)
        if val == 1 :
                step_cycle_matrix_y = [data]
        else:
                step_cycle_matrix_y = numpy.concatenate((step_cycle_matrix_y, [data]),axis=0)
    step_cycle_matrix_gt = []
    peak_height_gt = []
    #print(len(gtstance_peak))
    for val in range(1,len(gtstance_peak)):
            temp_range =y_gt[gtstance_peak[val-1]:gtstance_peak[val]]
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
    print(numpy.shape(numpy.mean(step_cycle_matrix_y,axis=0)))
    print(numpy.shape(numpy.mean(step_cycle_matrix_gt,axis=0)))
    plt.figure() 
    # fig.suptitle(subject)
    fig, ax = plt.subplots()
    #ax.fill_between(numpy.linspace(0,100,100),numpy.mean(step_cycle_matrix_y,axis=0)-numpy.std(step_cycle_matrix_y,axis=0), numpy.mean(step_cycle_matrix_y,axis=0)+numpy.std(step_cycle_matrix_y,axis=0), alpha=0.2,color='blue')
    ax.plot(numpy.linspace(0,100,100),numpy.mean(step_cycle_matrix_y,axis=0),c='b',label = 'IMU Estimate')
    # plt.plot(numpy.linspace(0,100,100),numpy.mean(step_cycle_matrix_y,axis=0)-numpy.std(step_cycle_matrix_y,axis=0),'b--',label = '+/- Standard Deviation')
    # plt.plot(numpy.linspace(0,100,100),numpy.mean(step_cycle_matrix_y,axis=0)+numpy.std(step_cycle_matrix_y,axis=0),'b--')
   
    #ax.fill_between(numpy.linspace(0,100,100),numpy.mean(step_cycle_matrix_gt,axis=0)-numpy.std(step_cycle_matrix_gt,axis=0), numpy.mean(step_cycle_matrix_gt,axis=0)+numpy.std(step_cycle_matrix_gt,axis=0), alpha=0.2,color='green')
    ax.plot(numpy.linspace(0,100,100),numpy.mean(step_cycle_matrix_gt,axis=0),c='b',label = 'Ground Truth')
    
    # plt.plot(numpy.linspace(0,100,100),numpy.mean(step_cycle_matrix_gt,axis=0),c='g',label = 'mean')
    # plt.plot(numpy.linspace(0,100,100),numpy.mean(step_cycle_matrix_gt,axis=0)-numpy.std(step_cycle_matrix_gt,axis=0),'g--',label = '+/- Standard Deviation')
    # plt.plot(numpy.linspace(0,100,100),numpy.mean(step_cycle_matrix_gt,axis=0)+numpy.std(step_cycle_matrix_gt,axis=0),'g--')
    plt.legend()
    fig.savefig(population+'.png')
    # axs[1].set_ylabel('Left '+str(round(np.mean(np.array(peak_height_l))))+' degrees')
    return
           

def train_one_epoch(net, optimizer, train_dloader, device, **kwargs):
    net.train()
    criterion = nn.MSELoss()
    loss_error = 0

    for _iter, batch in enumerate(train_dloader):
        x_in, y_gt,_ = batch
        #print(y_gt.size())
        
        # Concatenate left and right leg
        x_in = x_in.reshape(-1, *x_in.shape[-2:]).to(device)
        y_gr = y_gt.reshape(-1, *y_gt.shape[-2:]).to(device)

        # Predict
        y_pred = net(x_in)
        
        # Calculate loss
        loss = criterion(y_pred, y_gr)
        
        # y_pred = train_dloader.dataset.normalizer.unnormalize_output(y_pred.detach().cpu())
        # y_gr = train_dloader.dataset.normalizer.unnormalize_output(y_gr.cpu())

        #print(" Loss: {}".format(
        #     loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_error += loss.item()
    
    return loss_error / (_iter + 1)
    
    
def test_model(net, test_dloader, device,epoch,population):
    net.eval()
    criterion = nn.MSELoss()
    rmses = []

    with torch.no_grad():
        for _, batch in enumerate(test_dloader):
            x_in, y_gt,activity_label = batch
        
        # Concatenate left and right leg
            x_in = x_in.reshape(-1, *x_in.shape[-2:]).to(device)
            y_gr = y_gt.reshape(-1, *y_gt.shape[-2:]).to(device)
            y_pred = net.forward(x_in)
                        
            y_pred = test_dloader.dataset.normalizer.unnormalize_output(y_pred.detach().cpu())
            y_gr = test_dloader.dataset.normalizer.unnormalize_output(y_gr.detach().cpu())
            #x_temp = test_dloader.dataset.normalizer.unnormalize_output(x_in.detach().cpu())
            #plt.figure()
            #plt.plot(y_pred[1,:800,0])
            #plt.plot(y_gr[1,:800,0])
            #plt.savefig('Prediction'+str(_))
            #y_pred = y_pred - y_pred.mean(axis=1, keepdims=True) + y_gr.mean(axis=1, keepdims=True)
            #print(y_pred.size())
            #print(y_gr.size())
            rmse = ((y_pred - y_gr) ** 2).mean(1) ** 0.5
            #print(rmse)
            if rmse.detach().cpu()[1][0]>15 or rmse.detach().cpu()[0][0]>15 :
                plt.figure()
                plt.plot(y_pred[1,:800,0])
                plt.plot(y_gr[1,:800,0])
                plt.savefig(activity_label[0]+str(epoch))
            rmses.append(rmse)
            

            #plt.figure(figsize=(15, 15))
            #plt.plot(pred[0,:500,:].numpy().,label='Prediction')
            #plt.plot(y_gr[0,:500,:].numpy(),label='Ground Truth')
            #plt.savefig('result'+str(_iter)+'.png')

    rmses = torch.cat(rmses)
    

    
    #y_pred = y_pred - y_pred.mean(axis=1, keepdims=True) + y_gr.mean(axis=1, keepdims=True)
    rmse = ((y_pred - y_gr) ** 2).mean(1) ** 0.5
    
    #if epoch % 2 == 0:
        #plot_average_gait_cycle(y_pred[0,:,0].numpy(),y_gt[0,:,0].numpy())
    #if population == 'healthy' or population == 'acl':
        # y_gt = y_gt.squeeze(0)
        # y_gt = y_gt.squeeze(1)
        # print(y_gt.size())
        # print(y_pred.size())
    population = 'all'
    plot_average_gait_cycle(y_pred[0,:,0].numpy(),y_gr[0,:,0].numpy(),population)
    x=x_in.detach().cpu()
    
    plt.figure()
    plt.plot(y_pred[0,:500,0],label='prediction')
    plt.plot(y_gr[0,:500,0],label='GT')
    plt.plot(x_in.detach().cpu()[0,:500,0],label='IMU')
    plt.legend()
    plt.savefig('Prediction new'+str(epoch))
    return rmses.mean().item()
    
    
    

    
       
def main(**kwargs):
    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)
    device = 'cuda' if use_cuda else 'cpu'

    net, norm_dict = build_nn_model(**kwargs)
    net.to(device=device)

    optimizer = build_optimizer(net, **kwargs)

    train_dloader, test_dloader = setup_validation_data(norm_dict, device=device, **kwargs)

    loss_list = []
    rmse_list = []
    print(f'Before fine-tuning | Evaluation results: {test_model(net, test_dloader, device,0,False):.2f} (deg)')
    for epoch in range(1, kwargs.get('epochs') + 1):
        net_loss = train_one_epoch(net, optimizer, train_dloader, device, **kwargs)
        loss_list = numpy.append(loss_list,net_loss)
        eval_results = test_model(net, test_dloader, device,epoch,False)
        rmse_list = numpy.append(rmse_list,eval_results)
        print(f'Epoch {epoch:02d} | Evaluation results: {test_model(net, test_dloader, device,epoch,False):.2f} (deg)')
    torch.save(net.state_dict(),'model_dict_80_20.pt')
    
    
    
    # all_dloader,acl_dloader,healthy_dloader = setup_confirmation_data(norm_dict, device=device, **kwargs)
    # healthy = 'healthy'
    # acl = 'acl'
    # print(f'All RMSE: {test_model(net, all_dloader, device,epoch,False): .2f} (deg)')
    # print(f'Healthy RMSE: {test_model(net, healthy_dloader, device,epoch,healthy): .2f} (deg)')
    # print(f'ACL RMSE: {test_model(net, acl_dloader, device,epoch,acl): .2f} (deg)')

       


if __name__ == '__main__':
    args = parse_config()
    main(**args)