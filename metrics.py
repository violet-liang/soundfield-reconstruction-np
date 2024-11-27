import torch


def compute_NMSE(ref, pred):
    mse = torch.mean(torch.abs(ref.flatten(0,1) - pred.flatten(0,1)) ** 2, dim=0)
    nmse = mse / torch.mean(torch.abs(ref.flatten(0,1))**2, dim=0)
    return nmse#10*torch.log10(nmse)
    

def compute_MAC(ref, pref):
    mac=[]
    for i in range(ref.shape[-1]):
        ref1 = ref[:,:,i].flatten(0,1)
        pref1 = pref[:,:,i].flatten(0,1)
        a = torch.mm(pref1.unsqueeze(0).conj(),ref1.unsqueeze(-1))
        b = torch.mm(ref1.unsqueeze(0).conj(),ref1.unsqueeze(-1))*torch.mm(pref1.unsqueeze(0).conj(),pref1.unsqueeze(-1))
        zz = torch.square(a)/b
        mac.append(zz.squeeze())
    mac_all = torch.stack(mac, axis=0)       
    return mac_all
