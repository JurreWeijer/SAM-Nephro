import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight):
        super(DiceBCELoss, self).__init__()
        
        self.dice_weight = dice_weight

    def forward(self, outputs, target, smooth=1e-5):
        
        outputs = torch.sigmoid(outputs)

        # flatten label and prediction tensors
        outputs = outputs.view(-1)
        target = target.view(-1)
        
        intersection = (outputs * target).sum()
        dice_coefficient = (2.0 * intersection + smooth) / (outputs.sum() + target.sum() + smooth) 
        dice_loss = (1 - dice_coefficient)
        
        BCE = nn.functional.binary_cross_entropy(outputs, target, reduction="mean")
        
        return (1-self.dice_weight)*BCE + self.dice_weight*dice_loss

    