import torch
import logging
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F


def set_optimizer(model, params):
    """ Defines the optimizer to use during the training procedure """

    if params["optimizer"] == "Adam":
        logging.info('Using ' + params["optimizer"]  + ' as optimizer')

        return optim.Adam(model.parameters(), lr=params["initialLR"])

    elif params["optimizer"] == "SGD":
        logging.info('Using ' + params["optimizer"]  + ' as optimizer')

        return optim.SGD(model.parameters(), lr=params["initialLR"]) 

    elif params["optimizer"] == "RMSprop":
        logging.info('Using ' + params["optimizer"]  + ' as optimizer')

        return optim.RMSprop(model.parameters(), lr=params["initialLR"])       


def dice_loss(label, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this case, we would like to maximize 
    the dice loss so we return the negated dice loss.

    Args:
        label: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.

    Taken from: https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py#L78     
    """
    #print(label.shape, logits.shape)
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[label.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[label.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, label.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps))
    
    return [(1 - dice_loss.mean()), dice_loss.detach().cpu().numpy()]