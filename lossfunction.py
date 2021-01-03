####################################################
##### This is focal loss class for multi class #####
##### University of Tokyo Doi Kento            #####
####################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



# focal loss
class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def focal_loss_with_logits(self, logits, targets, y_pred):
        weight_a = self.alpha * (1 - y_pred) ** self.gamma * targets
        weight_b = (1 - self.alpha) * y_pred ** self.gamma * (1 - targets)

        return (torch.log1p(torch.exp(-torch.abs(logits))) + F.relu(
            -logits)) * (weight_a + weight_b) + logits * weight_b

    def forward(self, inputs, targets):
        y_pred = torch.clamp(inputs, torch.finfo(torch.float32).eps,
                                  1 - torch.finfo(torch.float32).eps)
        logits = torch.log(y_pred / (1 - y_pred))

        loss = self.focal_loss_with_logits(logits=logits, targets=targets,
                                           y_pred=y_pred)

        return torch.mean(loss)
    

    
# dice loss
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

    
    
# BCE-dice loss
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
    
    
    
# log cosh dice loss
# paper: A survey of loss functions for semantic segmentation
# https://arxiv.org/abs/2006.14822

class LogCoshDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LogCoshDiceLoss, self).__init__()
        
    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        # log cosh dice loss
        x = 1 - dice
        LCD_loss = torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)
        
        return LCD_loss
    
    
    
# focal tversky loss
# class FocalTverskyLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True, smooth=1, alpha=0.7, gamma=0.75):
#         super(FocalTverskyLoss, self).__init__()
#         self.smooth = smooth
#         self.alpha = alpha
#         self.gamma = gamma
        
#     def forward(self, inputs, targets):
#         pt_1 = self.tversky_index(inputs, targets)
#         return torch.pow((1 - pt_1), self.gamma)
    
#     def tversky_index(self, inputs, targets):
#         y_true_pos = torch.flatten(targets)
#         y_pred_pos = torch.flatten(inputs)
#         true_pos = torch.sum(y_true_pos * y_pred_pos)
#         false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
#         false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
#         return (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + (
#                     1 - self.alpha) * false_pos + self.smooth)
    
    
    
# focal tversky loss
class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky