import torch
import torch.nn as nn
import torch.nn.functional as F

class MT(nn.Module):
    def __init__(self, model, ema_factor):
        super().__init__()
        self.model = model
        self.model.train()
        self.ema_factor = ema_factor
        self.global_step = 0

    """
    def forward(self, x, y, model, mask):
        self.global_step += 1
        y_hat = self.model(x)
        model.update_batch_stats(False)
        y = model(x) # recompute y since y as input of forward function is detached
        model.update_batch_stats(True)
        return (F.mse_loss(y.softmax(1), y_hat.softmax(1).detach(), reduction="none").mean(1) * mask).mean()
    """

    def forward(self, x, y, model):
        self.global_step += 1
        y_hat = self.model(x)
        model.update_batch_stats(False)
        y = model(x) # recompute y since y as input of forward function is detached
        model.update_batch_stats(True)
        
        beta_coeff = 1
        alphas = torch.exp(y_hat)
        betas = torch.exp(y)
        a_zero = torch.sum(alphas, -1)
        
        """
        loss1 = torch.lgamma(a_zero) - torch.sum(torch.lgamma(alphas), -1)
        loss2 = torch.sum(
                    (alphas - betas) * (torch.digamma(alphas) - torch.digamma(a_zero.unsqueeze(-1))), -1)

        kl_loss = loss1 + loss2
        """
        kl_loss = torch.lgamma(a_zero) - torch.lgamma(torch.sum(betas,-1)) - torch.sum(torch.lgamma(alphas),-1) + \
                    torch.sum(torch.lgamma(betas),-1) + torch.sum((alphas-betas)*(torch.digamma(alphas) - torch.digamma(a_zero.unsqueeze(-1))),-1)
        
        
        return kl_loss.mean()
        #return (kl_loss * mask).mean()


    def moving_average(self, parameters):
        ema_factor = min(1 - 1 / (self.global_step+1), self.ema_factor)
        for emp_p, p in zip(self.model.parameters(), parameters):
            emp_p.data = ema_factor * emp_p.data + (1 - ema_factor) * p.data
