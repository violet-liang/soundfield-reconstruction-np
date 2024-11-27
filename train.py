import os
import argparse
import random
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from data import TrainDataset,TestDataset
from torch.utils.data import DataLoader
from network import LatentModel


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
  

def kl_div_gaussians(prior_mu, prior_var, posterior_mu, posterior_var):
    kl_div = (torch.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / torch.exp(prior_var) - 1. + (prior_var - posterior_var)
    kl_div = 0.5 * kl_div.sum()
    return kl_div


def np_loss(y_hat, y, post_mu, post_var, prior_mu, prior_var):
    L2 = torch.nn.functional.mse_loss(y_hat, y, reduction="mean") 
    KLD = kl_div_gaussians(post_mu, post_var, prior_mu, prior_var)
    return L2, KLD, L2 + KLD


def adjust_learning_rate(optimizer, step_num, warmup_step=1500):
    lr = 0.001 * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(optimizer, step_num, warmup_step=1500):
    if step_num < warmup_step:
        lr = 0.001 * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    elif step_num < 16000:
        lr = 0.001
    else:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    
def train(epoch, model, scheduler, optimizer, global_step, args):
    model.train()
    train_loss = 0
    for batch_idx, y_all in enumerate(train_loader):
        global_step += 1
        adjust_learning_rate(optimizer, global_step)

        batch_size = y_all.shape[0]
        y_all = y_all.to(device).view(batch_size, -1, 40)
        y_all_scale = (y_all-mean)/std

        N = 50
        context_idx = get_context_idx(N)
        context_idx =32*4*(context_idx//8) + 4*((context_idx%8)-1) + 1
        x_context = idx_to_x(context_idx, batch_size)
        y_context = idx_to_y(context_idx, y_all_scale)
        x_all = x_grid.expand(batch_size, -1, -1) 
       
        y_pred, prior_mu, prior_var, post_mu, post_var = model(x_context, y_context, x_all,  y_all_scale)
                
        l2loss, kldloss, loss = np_loss(y_pred, y_all_scale, post_mu, post_var, prior_mu, prior_var)
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            log = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tlr: {:.6f}\tL2 Loss: {:.6f}\tKLD Loss: {:.6f}, Total Loss: {:.6f}'.format(
                epoch, batch_idx * len(y_all), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       optimizer.state_dict()['param_groups'][0]['lr'],
                       l2loss.item(), kldloss.item(), loss.item())
            print(log)
            with open(args.exp_name+'/log.txt', 'a') as f:
                f.write(log+'\n')
    
    log = '====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader))
    print(log)
    with open(args.exp_name+'/log.txt', 'a') as f:
        f.write(log+'\n')
    torch.save(model.state_dict(), args.exp_name + '/%d.pth'%epoch)
    
    return global_step


def test(epoch, best_loss, model, args):
    model.eval()
    test_loss_all = 0
    with torch.no_grad():
        for i, y_all in enumerate(test_loader):
        
            y_all = y_all.to(device).view(y_all.shape[0], -1, 40)
            batch_size = y_all.shape[0]
            y_all_scale = (y_all-mean)/std

            N = 50
            context_idx = get_context_idx(N)
            context_idx =32*4*(context_idx//8) + 4*((context_idx%8)-1) + 1
            x_context = idx_to_x(context_idx, batch_size)
            y_context = idx_to_y(context_idx, y_all_scale)
            x_all = x_grid.expand(batch_size, -1, -1)
           
            y_pred, prior_mu, prior_var, post_mu, post_var = model(x_context, y_context, x_all)
            y_pred_unscale = y_pred*std+mean
            test_loss_all  +=  np_loss(y_pred_unscale, y_all, post_mu, post_var, prior_mu, prior_var)[0].item()
   
    
    test_loss_all /= len(test_loader) 
    
    if test_loss_all < best_loss:
        torch.save(model.state_dict(), args.exp_name+'/best_model.pth')
        best_loss = test_loss_all
    
    log = '====> Test set loss: {:.6f}'.format(test_loss_all) + '\n' + '====> Best loss: {:.6f}'.format(best_loss)
    print(log)
    with open(args.exp_name+'/log.txt', 'a') as f:
        f.write(log+'\n')
  
    return best_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Processes')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--exp_name', type=str, default='model-train',
                        help='the name of your experiment')             
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=226, metavar='S',
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
    

    train_dataset = TrainDataset('./train.npy')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = TrainDataset('./test.npy')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    #Decide whether to use standardization based on the data
    mean, std = train_dataset.get_mustd()
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)
    
    
    model = torch.nn.DataParallel(LatentModel(128,42)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    x_grid = generate_grid(32, 32)
  
    
    os.makedirs(args.exp_name, exist_ok=True)
    with open(args.exp_name+'/log.txt', 'w') as f:
        pass
    
    ## train=
    best_loss = 1000000
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        global_step = train(epoch, model, None, optimizer, global_step, args)
        best_loss = test(epoch, best_loss, model, args)
    
    
   
    
    
    
    
    
   