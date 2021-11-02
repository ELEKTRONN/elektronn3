import torch
import numpy as np

class MaskedLoss(torch.nn.Module):
    
    def __init__(self, weights_dist, channel_separation: list, loss: str = "L1"):
       

        super(MaskedLoss,self).__init__() 
        self.register_buffer('weights_dist', torch.as_tensor(weights_dist, dtype = torch.float32))
        self.channel_separation = channel_separation
        if loss == "L1":
            self.loss = torch.nn.L1Loss()
        elif loss == "L2":
            self.loss = torch.nn.MSELoss()
        else:
            raise NotImplementedError


    def forward(self, out, target):
        
        losslist = []
        for param, crange in zip(self.weights_dist, self.channel_separation):
            losslist.append(param * self.loss(out[:,crange[0] : crange[1]], target[:,crange[0]:crange[1]] ))
        loss = sum(losslist)
        return loss



#class StaticWeightedMSELoss(nn.Module):
#    """Weighted MSE loss where elements are weighted according to a weight mask."""
#    def __init__(self, weights) -> None:
#        super().__init__()
#        self.register_buffer('weights', torch.as_tensor(weights, dtype=torch.float32))
#
#    def forward(self, out, target):
#        err = F.mse_loss(out, target, reduction='none')
#        err *= self.weights
#        loss = err.mean()
#        return loss
#
#weights = <distance transform der binary mask, die dort 1 ist, wo im Input die Pixel immer auf 0 gesetzt werden. Überall sonst 0.>
#
#criterion = StaticMaskedMSELoss(weights=weights)
