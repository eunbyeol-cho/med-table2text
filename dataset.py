import torch
import torch.utils.data
import os
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from functools import reduce


class SNUHDataset(torch.utils.data.Dataset):
    def __init__(self, split, args):
        
        #Data args
        self.nc, self.nn, self.ntext = args.nc, args.nn, args.nt
        self.ntable = self.nc+self.nn
        self.text_vocab = args.text_vocab

        self.data_root = args.input_path
        self.mask_null = args.mask_null
        # self.split = split 

        self.split = 'test'

        self.seed = args.seed
        self.prefix  = 'snuh'   
        self.target = args.model.split('2')[-1]
        self.fold = os.path.join(self.data_root, "fold", "{}_{}_fold_split.csv".format(self.prefix, self.seed))  
        self.mlm_probability = args.valid_mlm_prob if (args.valid_mlm_prob is not None) and (args.eval) else args.mlm_prob
 
        #Load data
        hit_idcs = self.get_fold_indices()
        self.input_ids = np.load(os.path.join(self.data_root, 'input_ids.npy'))#[hit_idcs]
        self.token_type_ids = np.load(os.path.join(self.data_root, 'token_type_ids.npy'))[hit_idcs]
        self.null_type_labels = np.load(os.path.join(self.data_root, 'null_type_ids.npy'))[hit_idcs]

        cat_ids = self.input_ids[:, :self.nc]
        num_ids = self.input_ids[:, self.nc:self.ntable]
        text_ids = self.input_ids[:, self.ntable:]

        #Split train/valid/test
        self.cat_ids = cat_ids[hit_idcs]
        self.num_ids = num_ids[hit_idcs]
        self.text_ids = text_ids[hit_idcs]
        
        #MLM related args
        self.cat_null_token_id = args.cat_null_id
        self.num_null_token_id = args.num_null_id
        # self.mask_token_id = self.cat_null_token_id + 1
        # self.pad_token_id = self.cat_null_token_id + 2

        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.sep, self.pad, self.cls = tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.cls_token_id    
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id

        self.text_mask_ignore_token_ids = torch.LongTensor([self.sep, self.pad, self.cls])
        self.cat_mask_ignore_token_ids = None if self.mask_null else torch.LongTensor([self.cat_null_token_id]) 
        self.num_mask_ignore_token_ids = None if self.mask_null else torch.FloatTensor([self.num_null_token_id]) #nan

    
    def __len__(self):
        return len(self.cat_ids)
        

    def __getitem__(self, index):

        cat_ids = torch.LongTensor(self.cat_ids[index])
        num_ids = torch.FloatTensor(self.num_ids[index])
        text_ids = torch.LongTensor(self.text_ids[index])
        token_type_ids = torch.LongTensor(self.token_type_ids[index])
        null_type_ids = torch.LongTensor(self.null_type_labels[index])

        cat_labels = cat_ids.clone()
        num_labels = num_ids.clone()
        null_type_labels = null_type_ids.clone()

        assert self.target == 'text', 'Target should be text.'
        text_ids, text_labels = self.mask_tokens(text_ids, self.text_mask_ignore_token_ids)
        
        return {
            'cat_ids' : cat_ids,
            'cat_labels' : cat_labels,
            'num_ids' : num_ids,
            'num_labels' : num_labels,
            'text_ids' : text_ids,
            'text_labels':text_labels,
            'token_type_ids' : token_type_ids,
            'null_type_ids' : null_type_ids,
            'null_type_labels' : null_type_labels,
        }
    

    def get_fold_indices(self):

        if self.split == 'train':
            hit = 1
        elif self.split == 'valid':
            hit = 2
        elif self.split == 'test':
            hit = 0

        df = pd.read_csv(self.fold)
        splits = df['fold'].values
        idcs = np.where(splits == hit)[0]

        return idcs


    def mask_tokens(self, inputs, special_tokens_masks, fixed_mask_indices=None):
        """
        Ref: https://huggingface.co/transformers/v4.9.2/_modules/transformers/data/data_collator.html
        """
        
        inputs = inputs.unsqueeze(0)
        labels = inputs.clone()
        init_no_mask = torch.full_like(inputs, False, dtype=torch.bool)

        if special_tokens_masks is not None:

            special_tokens_mask = reduce(lambda acc, el: acc | (inputs == el), special_tokens_masks, init_no_mask)
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            special_tokens_mask = special_tokens_mask.bool()

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        else : 
            special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
            probability_matrix = torch.full(labels.shape, self.mlm_probability)

        #Fix mask indices
        if fixed_mask_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()
        else: 
            masked_indices = torch.unsqueeze(torch.BoolTensor(fixed_mask_indices), dim=0)
            
        labels[~masked_indices] = -100
        inputs[masked_indices] = self.mask_token_id
        
        return inputs.squeeze(0), labels.squeeze(0)


    def collator(self, samples):

        samples = [s for s in samples if s["cat_ids"] is not None]
        if len(samples) == 0:
            return {}
        
        input = {"cat_ids": torch.stack([s["cat_ids"] for s in samples])}
        input["num_ids"] = torch.stack([s["num_ids"] for s in samples])
        input["text_ids"] = torch.stack([s["text_ids"] for s in samples])
        
        input["cat_labels"] = torch.stack([s["cat_labels"] for s in samples])
        input["num_labels"] = torch.stack([s["num_labels"] for s in samples])
        input["text_labels"] = torch.stack([s["text_labels"] for s in samples])
        
        input["token_type_ids"] = torch.stack([s["token_type_ids"] for s in samples])
        input["null_type_ids"] = torch.stack([s["null_type_ids"] for s in samples])
        input["null_type_labels"] = torch.stack([s["null_type_labels"] for s in samples])

        out = {}
        out["net_input"] = input

        return out