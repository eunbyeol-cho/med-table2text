import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

class TextCriterion(_Loss):
    def __init__(self, args):
        super().__init__()
        
        self.text_criterion = nn.NLLLoss(ignore_index=0)
        self.total_loss = 0
        

    def forward(self, net_output):

        text_labels = net_output['text_labels'] 
        text_pred = net_output['text_pred'] 
        
        text_mask = (text_labels == -100)
        text_labels = text_labels[~text_mask]
        text_pred = text_pred[~text_mask]

        loss = self.text_criterion(text_pred, text_labels)
        self.total_loss += loss.item()

        return {
            'loss' : loss,
            'total_loss' : self.total_loss,
        }


class TextMetric:
    def __init__(self, args):
        
        self.text_acc = 0    

    def __call__(self, net_output):
        return self.get_accuracy(net_output)

    def get_accuracy(self, net_output):

        text_labels = net_output['text_labels'] 
        text_pred = net_output['text_pred'] 

        text_mask = (text_labels == -100)
        text_labels = text_labels[~text_mask]
        text_pred = text_pred[~text_mask]
        
        text_pred_argmax = torch.argmax(text_pred, dim=1)
        correct = (text_labels==text_pred_argmax).sum().detach().cpu()
        self.text_acc += int(correct)/(len(text_labels)) 

        return {
            'text_acc':self.text_acc
        }
