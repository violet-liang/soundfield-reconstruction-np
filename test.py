import os
import argparse
import random
import torch
import torch.utils.data
from torch import nn, optim
from data import TrainDataset,TestDataset
from torch.utils.data import DataLoader
from network import LatentModel
import numpy as np
import matplotlib.pyplot as plt
from metrics import *


def get_context_idx(N):
    idx = random.sample(range(1, 64), N)
    idx = torch.tensor(idx, device=device)
    return idx


def generate_grid(h, w):
    rows = torch.linspace(0, 1, h, device=device) 
    cols = torch.linspace(0, 1, w, device=device)
    grid = torch.stack([cols.repeat(h, 1).t().contiguous().view(-1), rows.repeat(w)], dim=1)
    grid = grid.unsqueeze(0)
    return grid


def idx_to_y(idx, data):
    y = torch.index_select(data, dim=1, index=idx)
    return y


def idx_to_x(idx, batch_size):
    x = torch.index_select(x_grid, dim=1, index=idx)
    x = x.expand(batch_size, -1, -1)
    return x


def kl_div_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    var_p = torch.exp(logvar_p)
    kl_div = (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / var_p \
             - 1.0 \
             + logvar_p - logvar_q
    kl_div = 0.5 * kl_div.sum()
    return kl_div


def np_loss(y_hat, y, post_mu, post_var, prior_mu, prior_var):
    L2 = torch.nn.functional.mse_loss(y_hat, y, reduction="mean") 
    KLD = kl_div_gaussians(post_mu, post_var, prior_mu, prior_var)
    return L2, KLD, L2 + KLD

    
def test(epoch, best_loss, model, args):
    model.eval()
    test_loss_all = 0
    total_nmse = 0
    total_mac = 0
    with torch.no_grad():
        for i, (y_all) in enumerate(test_loader):
           
            y_all = y_all.to(device).view(y_all.shape[0], -1, 40)
            batch_size = y_all.shape[0]
            y_all_scale = (y_all-mean)/std

            N =50
            context_idx = get_context_idx(N)
            context_idx =32*4*(context_idx//8) + 4*((context_idx%8)-1) + 1
            x_context = idx_to_x(context_idx, batch_size)
            y_context = idx_to_y(context_idx, y_all_scale)
            x_all = x_grid.expand(batch_size, -1, -1)
        
            y_pred, prior_mu, prior_var, post_mu, post_var = model(x_context, y_context, x_all)
            y_pred_origin = y_pred*std+mean
            test_loss_all  +=  np_loss(y_pred_origin, y_all, post_mu, post_var, prior_mu, prior_var)[0].item()
    
            total_nmse += compute_NMSE(y_all, y_pred_origin) #B,S,40
            total_mac += compute_MAC(y_all, y_pred_origin)   
           
    average_nmse = total_nmse/(i+1)    
    average_mac = total_mac/(i+1)
    print(10*torch.log10(average_nmse), average_mac.real, sep='\n')

    test_loss_all /= len(test_loader) 
    print('====> Test set loss: {:.4f}'.format(test_loss_all))
    print('====> Best loss: {:.4f}'.format(best_loss))
  
    return best_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Processes')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--r_dim', type=int, default=128, metavar='N',
                        help='dimension of r, the hidden representation of the context points')
    parser.add_argument('--z_dim', type=int, default=128, metavar='N',
                        help='dimension of z, the global latent variable')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    train_dataset = TrainDataset('/home/znliang/ANP/rir_matlab/ntrain500hz-1w.npy')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = TrainDataset('/home/znliang/ANP/rir_matlab/ntest500hz.npy')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    mean, std = train_dataset.get_mustd()
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)
    
    model = torch.nn.DataParallel(LatentModel(128,42)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    x_grid = generate_grid(32, 32)
    os.makedirs("results/", exist_ok=True)
    
    best_loss = 1000000
    model.load_state_dict(torch.load('./model-train/best_model.pth'))
    model.eval()
    best_loss = test(300, best_loss, model, args)
  
   
    
    
    
    
    
   