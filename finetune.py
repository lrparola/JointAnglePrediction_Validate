from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from configs.cmd_parse import parse_config
from lib.data.dloader import setup_validation_data
from lib.models.builder import build_nn_model, build_optimizer

import torch
import torch.nn as nn
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import torch
import numpy
from pdb import set_trace as st


def train_one_epoch(net, optimizer, train_dloader, device, **kwargs):
    net.train()
    criterion = nn.MSELoss()
    loss_error = 0

    for _iter, batch in enumerate(train_dloader):
        x_in, y_gt = batch
        
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
    
    
def test_model(net, test_dloader, device):
    net.eval()
    criterion = nn.MSELoss()
    rmses = []

    with torch.no_grad():
        for _, batch in enumerate(test_dloader):
            x_in, y_gt = batch
        
        # Concatenate left and right leg
            x_in = x_in.reshape(-1, *x_in.shape[-2:]).to(device)
            y_gr = y_gt.reshape(-1, *y_gt.shape[-2:]).to(device)
            y_pred = net.forward(x_in)
                        
            y_pred = test_dloader.dataset.normalizer.unnormalize_output(y_pred.detach().cpu())
            y_gr = test_dloader.dataset.normalizer.unnormalize_output(y_gr.detach().cpu())

            rmse = ((y_pred - y_gr) ** 2).mean(1) ** 0.5
            rmses.append(rmse)

            #plt.figure(figsize=(15, 15))
            #plt.plot(pred[0,:,:].numpy())
            #plt.plot(y_gr[0,:,:].numpy())
            #plt.savefig('result'+str(_iter)+'.png')

    rmses = torch.cat(rmses)
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
    print(f'Before fine-tuning | Evaluation results: {test_model(net, test_dloader, device):.2f} (deg)')
    for _ in range(1, kwargs.get('epochs') + 1):
        net_loss = train_one_epoch(net, optimizer, train_dloader, device, **kwargs)
        loss_list = numpy.append(loss_list,net_loss)
        eval_results = test_model(net, test_dloader, device)
        print(f'Epoch {_:02d} | Evaluation results: {test_model(net, test_dloader, device):.2f} (deg)')
    
    plt.figure(figsize=(15, 15))
    plt.plot(numpy.linspace(0,len(loss_list),len(loss_list)),loss_list)

    plt.title('Training Loss Verse Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss') 
    plt.savefig('loss.png')

    st()    


if __name__ == '__main__':
    args = parse_config()
    main(**args)
