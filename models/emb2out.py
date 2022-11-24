import torch
from torch import nn
import numpy as np
import os, sys, copy
from transformers import BertModel, BertTokenizer
from models import register_model
from utils import *

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

@register_model("emb2table")
class Emb2table(nn.Module):
    def __init__(self, args):
        super().__init__()

        #Model
        self.test = args.eval
        self.emb_dim = 256

        #Data
        self.nc = args.nc
        self.nn = args.nn
        self.ntable = self.nc+self.nn
        self.ntext = args.nt
        
        self.cat_vocab_size = args.cat_vocab_size
        self.text_vocab = args.text_vocab
        cat_nunique = args.cat_columns_numclass_list
        
        ##1. Null classifier
        self.null_classifier = nn.Linear(self.emb_dim, 2) if args.mask_null else None

        ##2-1. Numeric Regressor
        self.sep_regressor = args.sep_regressor
        self.mean_head = nn.Parameter(torch.randn(self.nn, self.emb_dim)) if self.sep_regressor else nn.Linear(self.emb_dim, 1)
        self.var_head = nn.Parameter(torch.randn(self.nn, self.emb_dim)) if self.sep_regressor else nn.Linear(self.emb_dim, 1)
        
        ##2-2. Categorical Classifier
        self.classifier = nn.Linear(self.emb_dim, self.cat_vocab_size) 
        self.softmax_mask = torch.full((args.nc, args.cat_vocab_size), -100000).to('cuda')
        cumulated = 0
        for s, v in enumerate(cat_nunique):
            self.softmax_mask[s, cumulated:cumulated+v] = 0
            cumulated += v
        self.softmax_layer = nn.LogSoftmax(dim=-1)

    
    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)


    def forward(self, x, **kwargs):

        ##1. Null classifier
        null_pred = self.null_classifier(x[:, :self.ntable]) if self.null_classifier is not None else None

        ##2-1. Categorical Classifier
        cat_pred = self.classifier(x[:, :self.nc])
        #Add mask & softmax layer if test
        cat_pred = self.softmax_layer(cat_pred + self.softmax_mask) if self.test else self.softmax_layer(cat_pred) 

        ##2-2. Numeric Regressor
        if self.sep_regressor:
            mean_pred = torch.sum(torch.mul(x[:, self.nc:self.ntable], self.mean_head), axis=-1).unsqueeze(-1) 
            var_pred = torch.sum(torch.mul(x[:, self.nc:self.ntable], self.var_head), axis=-1).unsqueeze(-1)
        else:
            mean_pred = self.mean_head(x[:, self.nc:self.ntable])
            var_pred = self.var_head(x[:, self.nc:self.ntable])

        return {
            'cat_labels' : kwargs['cat_labels'],
            'num_labels' : kwargs['num_labels'],
            'null_type_labels': kwargs['null_type_labels'],
            'null_type_ids': kwargs['null_type_ids'],
            'cat_pred':cat_pred,
            'mean_pred': mean_pred,
            'var_pred': var_pred,
            'null_pred': null_pred
        }



@register_model("emb2text")
class Emb2text(nn.Module):
    def __init__(self, args):
        super().__init__()

        #Model
        self.test = args.eval
        self.emb_dim = 256

        #Data
        self.ntable = args.nc+ args.nn
        self.ntext = args.nt
        self.text_vocab = args.text_vocab
        
        ##Text Classifier
        self.classifier = nn.Linear(self.emb_dim, self.text_vocab) 
        self.softmax_layer = nn.LogSoftmax(dim=-1)

    
    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)


    def forward(self, x, **kwargs):
        
        text_pred = self.softmax_layer(self.classifier(x[:, self.ntable:]))

        return {
            'text_labels' : kwargs['text_labels'],
            'text_pred':text_pred,
        }



@register_model("BERT")
class BERT(nn.Module):
    def __init__(self, args):
        super().__init__()

        len_input = 1000
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertModel.from_pretrained("prajjwal1/bert-mini")
        model = self.customize_model(model, len_input, tokenizer.vocab_size)
        self.encoder = model.encoder


    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)        


    def forward(self, x):
        x = self.encoder(x)
        return x.last_hidden_state
    

    def customize_model(self, model, len_input, vocab_size):
        #Change config and reinitialize the model (not just modifying the instance)
        model.config.__dict__['max_position_embeddings'] = len_input
        model.config.__dict__['vocab_size'] = vocab_size
        customized_model = BertModel(model.config)
        return customized_model