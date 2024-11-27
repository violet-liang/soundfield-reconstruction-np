from module import *

class LatentModel(nn.Module):
   
    def __init__(self, num_hidden, input_dim=42):
        super(LatentModel, self).__init__()
        self.gp_encoder = GPEncoder(num_hidden, num_hidden, input_dim=input_dim)
        self.kernel_encoder = KernelEncoder(num_hidden, num_hidden, input_dim=input_dim)
        self.decoder = Decoder(num_hidden, input_dim=input_dim-2)
        
    def forward(self, context_x, context_y, target_x, target_y=None, room_all=None, prior_time=None):

        num_targets = target_x.size(1)
        prior_mu, prior_var, prior = self.gp_encoder(context_x, context_y, room_all)
        
        # For training
        if target_y is not None:
            posterior_mu, posterior_var, posterior = self.gp_encoder(target_x, target_y, room_all)
            z = posterior
        
        # For Generation
        else:
            posterior_mu = prior_mu
            posterior_var = prior_var
            z = prior
        
        z = z.unsqueeze(1).repeat(1,num_targets,1) # [B, T_target, D]
        r, attns = self.kernel_encoder(context_x, context_y, target_x, room_all) # [B, T_target, D]
        
        y_pred = self.decoder(r, z, target_x)
        
        return y_pred, prior_mu, prior_var, posterior_mu, posterior_var
    
    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        kl_div = (t.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / t.exp(prior_var) - 1. + (prior_var - posterior_var)
        kl_div = 0.5 * kl_div.sum()
        return kl_div