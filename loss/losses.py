import torch.nn as nn
import torch

class Regloss_MSE(nn.Module):
    def __init__(self):
        super(Regloss_MSE, self).__init__()
    
    def forward(self, y_pred, y_true):
        #print (y_pred.shape, y_true.shape)
        no_object_loss = torch.pow((1 - y_true[:, 0]) * y_pred[:, 0],2).mean()
        object_loss = torch.pow((y_true[:, 0]) * (y_pred[:, 0] - 1),2).mean()

        reg_loss = (y_true[:, 0] * (torch.pow(y_true[:, 1:5] - y_pred[:, 1:5],2).sum(1))).mean()    
        
        loss = no_object_loss + object_loss + reg_loss
        return loss


class Regloss_SmoothL1(nn.Module):
    def __init__(self):
        super(Regloss_SmoothL1, self).__init__()
        self._reg_loss = nn.SmoothL1Loss()

    def forward(self, y_pred, y_true):
        #print (y_pred.shape, y_true.shape)
        no_object_loss = torch.pow((1 - y_true[:, 0]) * y_pred[:, 0],2).mean()
        object_loss = torch.pow((y_true[:, 0]) * (y_pred[:, 0] - 1),2).mean()

        mask = y_true[:,0].view(-1, 1).expand(y_true.size(0), 4)
        reg_loss = self._reg_loss(y_pred[:, 1:5]*mask, y_true[:, 1:5]*mask)

        loss = no_object_loss + object_loss + reg_loss
        return loss

        

        
    
    
