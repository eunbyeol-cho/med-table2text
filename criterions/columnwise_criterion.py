import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import pandas as pd
import os

class ColumnwiseCriterion(_Loss):
    def __init__(self, args):
        super().__init__()

        #Data args
        self.args = args

        #Initialize loss & acc
        self.text_acc = torch.zeros(args.nt) 
        self.text_count = torch.zeros(args.nt) + 1e-10
            
        #Save model output for the inference & visualization steps
        self.total_pred = torch.FloatTensor([]).cuda()
        self.total_labels = torch.FloatTensor([]).cuda()


    def forward(self, net_output):

        text_labels = net_output['text_labels'] 
        text_pred = net_output['text_pred'] 
        
        text_labels_full_shape = text_labels.float().clone()
        text_pred_prob = nn.Softmax(dim=-1).forward(text_pred)
        text_pred_full_shape = torch.distributions.Categorical(text_pred_prob).sample().float() # sampling using Softmax output

        text_mask = (text_labels == -100) 
        text_correct_full_shape = torch.zeros_like(text_labels).to(text_labels.device)
        text_correct_full_shape[~text_mask] += (text_labels == text_pred_full_shape)[~text_mask]

        self.text_acc += text_correct_full_shape.sum(axis=0).cpu()
        self.text_count += (~text_mask).sum(axis=0).cpu()
        
        self.total_pred = torch.cat((self.total_pred, text_pred_full_shape), axis=0)
        self.total_labels = torch.cat((self.total_labels, text_labels_full_shape), axis=0)

        return