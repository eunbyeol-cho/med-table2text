import torch
import torch.nn as nn
import logging
from models import register_model, MODEL_REGISTRY

logger = logging.getLogger(__name__)

class GenModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.src_modality, self.trg_modality = args.model.split('2')
        self.input2emb_model = self._input2emb_model.build_model(args) 
        self.encode_model = self._encode_model.build_model(args)
        self.emb2out_model = self._emb2out_model.build_model(args)
        
    @property
    def _input2emb_model(self):
        return MODEL_REGISTRY[f'{self.src_modality}2emb']

    @property
    def _encode_model(self):
        return MODEL_REGISTRY['BERT']

    @property
    def _emb2out_model(self):
        return MODEL_REGISTRY[f'emb2{self.trg_modality}']

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        return cls(args)


    def forward(self, **kwargs): 
        x = self.input2emb_model(**kwargs)
        x = self.encode_model(x)
        x = self.emb2out_model(x, **kwargs)
        return x