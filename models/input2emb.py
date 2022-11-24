import torch
import torch.nn as nn
import math, os
import numpy as np

from models import register_model

class BaseEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.nc = args.nc #num_categorical
        self.nn = args.nn #num_numerical
        self.ntable = self.nc+self.nn #num_table
        self.ntext = args.nt #num_text

        self.cat_vocab_size = args.cat_vocab_size
        self.text_vocab = args.text_vocab
        self.emb_dim = args.emb_dim
        self.dropout = 0.2

        """
        1. Embedding
        - Categorical EMbedding
        - Numeric Embedding
        - Text Embedding
        """
        self.categorical_emb = nn.Embedding(self.cat_vocab_size, self.emb_dim)
        self.sep_numemb = args.sep_numemb

        if self.sep_numemb:
            self.numeric_emb = nn.Parameter(torch.randn(self.nn, self.emb_dim))
        else:
            self.numeric_emb = nn.Linear(1, self.emb_dim)
        
        self.text_Emb = nn.Embedding(self.text_vocab, self.emb_dim, padding_idx=0)
        
        """
        2. Positional Encoding
        - Table: Column-wise Learnable Embedding
        - Text: Sinusoidal Positional Encoding
        """
        self.columnwise_emb = nn.Parameter(torch.randn(self.ntable, self.emb_dim))
        self.sinusoidal_emb = PositionalEncoding(self.emb_dim, self.dropout, self.ntext)

        """
        3. Token type Embedding: Table vs Text
        """
        self.token_type_emb = nn.Embedding(2, self.emb_dim)
        
        """
        4. Null type Embedding
        - 0: null
        - 1: not null
        - 2: unknown
        """
        null_type = 3 if args.mask_null else 2
        self.null_indicator = nn.Embedding(null_type, self.emb_dim)
        

    def forward(cat_ids, num_ids, text_ids, token_type_ids, null_type_ids, **kwargs):
        raise NotImplementedError()



@register_model("both2emb")
class Both2TableEmbedding(BaseEmbedding):

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def forward(self, cat_ids, num_ids, text_ids, token_type_ids, null_type_ids, **kwargs):
        
        #1. Apply embedding separately
        cat_ids = self.categorical_emb(cat_ids)
        text_ids = self.text_Emb(text_ids)
        
        if self.sep_numemb: #element-wise multiplication
            num_ids = num_ids.unsqueeze(-1)
            B, S, _ = num_ids.shape
            num_ids = torch.mul(num_ids.expand(B, S, self.emb_dim), self.numeric_emb)
        else : 
            num_ids = self.numeric_emb(num_ids.unsqueeze(-1))

        input_ids = torch.cat([cat_ids, num_ids, text_ids], axis=1)
        
        #2. Positional embedding
        ## Column-wise Learnable Embedding
        input_ids[:, :self.ntable] += self.columnwise_emb

        ## Sinusoidal Positional Encoding
        pe_index = torch.arange(0, self.ntext).expand(input_ids.shape[0], self.ntext).to(input_ids.device, dtype=torch.long)
        input_ids[:, self.ntable:] = self.sinusoidal_emb(input_ids[:, self.ntable:], pe_index) 

        #3. Token type embedding
        input_ids += self.token_type_emb(token_type_ids) 

        #4. Null type embedding
        input_ids[:,:self.ntable] += self.null_indicator(null_type_ids) 

        return input_ids



@register_model("table2emb")
class Table2TableEmbedding(BaseEmbedding):

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)

    def forward(self, cat_ids, num_ids, text_ids, token_type_ids, null_type_ids, **kwargs):
                       
        #1. Apply embedding separately
        cat_ids = self.categorical_emb(cat_ids)

        if self.sep_numemb: #element-wise multiplication
            B, S  = num_ids.shape
            num_ids = num_ids.unsqueeze(-1)
            num_ids = torch.mul(num_ids.expand(B, S, self.emb_dim), self.numeric_emb)
        else:
            num_ids = self.numeric_emb(num_ids.unsqueeze(-1))   
    
        input_ids = torch.cat([cat_ids, num_ids], axis=1)

        #2. Positional embedding - Column-wise Learnable Embedding
        input_ids[:, :self.ntable] += self.columnwise_emb

        #3. Token type embedding (not needed cuz only table case)
        # input_ids += self.token_type_emb(token_type_ids)

        #4. Null type embedding
        input_ids[:,:self.ntable] += self.null_indicator(null_type_ids)

        return input_ids



@register_model("text2emb")
class Table2TableEmbedding(BaseEmbedding):


    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)


    def forward(self,  text_ids, **kwargs):

        input_ids = self.text_Emb(text_ids)
        pe_index = torch.arange(0, self.ntext).expand(input_ids.shape[0], self.ntext).to(input_ids.device, dtype=torch.long)
        input_ids = self.sinusoidal_emb(input_ids, pe_index) 
        raise input_ids



class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dimension, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        max_len = int(max_len)

        pe = torch.zeros(max_len, embedding_dimension)   
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dimension, 2).float() * (-math.log(10000.0) / embedding_dimension))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.positional_embed = nn.Embedding(int(max_len), embedding_dimension, _weight=pe)
        
    def forward(self, x, pe_index):

        with torch.no_grad():
            positional_embed = self.positional_embed(pe_index) #(B, S, E)
        x = x + positional_embed 

        output = self.dropout(x)
        return output

    

