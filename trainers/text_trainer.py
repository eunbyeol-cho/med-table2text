import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel.data_parallel import DataParallel
import tqdm, os, pickle, wandb, csv, logging, pprint
import numpy as np
import pandas as pd 

import models
from dataset import SNUHDataset
from utils.trainer_utils import rename_logger,should_stop_early,load_model,load_optimizer, log_from_dict
from criterions.text_criterion import TextCriterion, TextMetric
from criterions.columnwise_criterion import ColumnwiseCriterion

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args):
        
        self.args = args

        #Data args
        self.input_path = args.input_path
        self.norm_type = args.sub_root
        self.fold = args.fold
        self.seed = args.seed
        self.num_null_id = args.num_null_id
        self.pad_token_id = args.cat_null_id + 2
        self.unnorm = args.unnorm

        self.valid_subsets = args.valid_subsets
        self.batch_size = args.batch_size
        self.data_loaders = dict()

        for subset in ['train'] + self.valid_subsets:
            self.load_dataset(args, subset)    

        #Mask related args
        self.mask_null = args.mask_null
        self.mlm_prob = args.mlm_prob
        self.valid_mlm_prob = args.valid_mlm_prob
        self.null_classifier = args.null_classifier

        #Model args
        self.start_epoch = 1
        model = models.generator.GenModel(args)
        print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))   
        
        if args.resume is not None:
            epochs, model = load_model(os.getcwd(), model)
            self.start_epoch = epochs
            print("Resume training from {} epoch".format(epochs))   
        self.model = nn.DataParallel(model, device_ids=args.device_ids).to('cuda')

        #Training args
        self.n_epochs = args.n_epochs
        self.lr = args.lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if args.resume is not None:
            self.optimizer = load_optimizer(os.getcwd(), self.optimizer)  

        self.patience = args.patience
        self.save_dir = args.save_dir
        self.save_prefix = args.save_prefix

        logger.info(pprint.pformat(vars(args)))      

        #Wandb init
        if not self.args.debug:
            wandb.init(
                project=self.args.wandb_project_name,
                entity="emrsyn",
                config=self.args,
                reinit=True
            )
            wandb.run.name = self.args.wandb_run_name


    def load_dataset(self, args, split:str):
        
        shuffle_flag = True if split == 'train' else False
        dataset = SNUHDataset(split, args)
        self.data_loaders[split] = DataLoader(
            dataset, collate_fn=dataset.collator, batch_size=self.batch_size, num_workers=8, shuffle=shuffle_flag
        )


    def train(self):

        self.model.train()

        for epoch in range(self.start_epoch, self.n_epochs + 1):
            logger.info(f'trainer loop start {epoch}')

            #Initialize all loss, acc every epoch
            self.criterion = TextCriterion(self.args)
            self.metric = TextMetric(self.args)

            for sample in tqdm.tqdm(self.data_loaders['train']):
                
                self.optimizer.zero_grad(set_to_none=True)

                net_output = self.model(**sample['net_input'])

                loss_dict = self.criterion(net_output)
                loss_dict['loss'].backward()
                self.optimizer.step()
                
                with torch.no_grad():
                    acc_dict = self.metric(net_output)
            
            #Wandb log
            del loss_dict['loss']
            loss_dict.update(acc_dict)
            

            for key in loss_dict:    
                loss_dict[key] /= len(self.data_loaders['train'])
            temp_wandb_log = dict(map(lambda kv: ('train_'+kv[0], (kv[1])), loss_dict.items()))
            wandb_log = log_from_dict(temp_wandb_log, 'train', epoch)

            if not self.args.debug:
                wandb.log(wandb_log)

            should_stop = self.validate_and_save(epoch, subset='valid')
            if should_stop:
                break

        #Test    
        if 'test' in self.args.valid_subsets:
            self.test(epoch, subset='test', load_checkpoint=self.args.load_checkpoint)
            print(f'test finished at epoch {epoch}')
        
        if self.args.debug == False:
            wandb.finish(0)


    def validate(self, epoch, subset):

        self.model.eval()

        with torch.no_grad():
            logger.info("begin validation")

            #Initialize all loss, acc every epoch
            self.criterion = TextCriterion(self.args)
            self.metric = TextMetric(self.args)
            valid_accs= []
            
            for sample in tqdm.tqdm(self.data_loaders[subset]):

                net_output = self.model(**sample['net_input'])
                loss_dict = self.criterion(net_output)
                acc_dict = self.metric(net_output)

            #Wandb log
            del loss_dict['loss']
            loss_dict.update(acc_dict)

            for key in loss_dict:    
                loss_dict[key] /= len(self.data_loaders[subset])
            temp_wandb_log = dict(map(lambda kv: (f'{subset}_'+kv[0], (kv[1])), loss_dict.items()))  
            wandb_log = log_from_dict(temp_wandb_log, subset, epoch)

            if not self.args.debug:
                wandb.log(wandb_log)
                            
            valid_accs.append(-loss_dict['total_loss'])

        return valid_accs

        
    def validate_and_save(self, epoch, subset):

        should_stop = False
        valid_acc = self.validate(epoch, subset)
        should_stop |= should_stop_early(self.patience, valid_acc[0])

        prev_best = getattr(should_stop_early, "best", None)
        if (
            self.patience <= 0
            or prev_best is None
            or (prev_best and prev_best == valid_acc[0])
        ):

            print("Saving checkpoint to {}".format(os.path.join(self.save_dir, self.save_prefix + "_best.pt")))

            torch.save(
                {
                    'model_state_dict': self.model.module.state_dict() if (
                        isinstance(self.model, DataParallel)
                    ) else self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epochs': epoch,
                    'args': self.args,
                },
                os.path.join(self.save_dir, self.save_prefix + "_best.pt")
            )

            print("Finished saving checkpoint to {}".format(os.path.join(self.save_dir, self.save_prefix + "_best.pt")))

        return should_stop


    def test(self, epoch, subset, load_checkpoint=None):

        self.model.test = True
        logger.info("begin test")
        
        if load_checkpoint is None:
            load_path = os.path.join(self.save_dir, self.save_prefix + "_best.pt")
        else: 
            load_path = load_checkpoint

        state_dict = torch.load(load_path, map_location='cpu')['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'module' in k:
                k = k.repalce('module.', '')
            new_state_dict[k] = v
            
        self.model.load_state_dict(new_state_dict, strict = True)
        return self.validate(epoch, subset)


    def generate(self):
        
        epoch = self.start_epoch
        valid_subsets = self.valid_subsets
        
        self.model.eval()
        #Initialize all loss, acc every epoch
        self.criterion = TextCriterion(self.args)
        self.metric = TextMetric(self.args)
        self.columnwise_criterion = ColumnwiseCriterion(self.args)

        #Iteration starts        
        for subset in valid_subsets:
            logger.info("begin validation on '{}' subset".format(subset))

            for sample in tqdm.tqdm(self.data_loaders[subset]):

                with torch.no_grad():

                    net_output = self.model(**sample['net_input'])
                    loss_dict = self.criterion(net_output)
                    acc_dict = self.metric(net_output)
                    self.columnwise_criterion(net_output)

        #Wandb log
        del loss_dict['loss']
        loss_dict.update(acc_dict)

        count = 0
        for subset in valid_subsets:
            count += len(self.data_loaders[subset])

        for key in loss_dict:
            loss_dict[key] /= count

        with rename_logger(logger, subset):
            logger.info(
                f"epoch: {epoch},\
                loss: {loss_dict['total_loss']:.3f},\
                text_acc: {loss_dict['text_acc']:.3f}"
            )    
        
        if not self.args.debug:
            wandb.log(loss_dict)

        #Save generated samples
        pred_csv_path = os.path.join(os.getcwd(), f'columnwise_pred_{self.norm_type}_{self.args.model}_{self.mlm_prob}_{self.valid_mlm_prob}_{self.seed}_{self.null_classifier}.csv')
        df_pred_output = pd.DataFrame(self.columnwise_criterion.total_pred.cpu().numpy())
        df_pred_output.to_csv(pred_csv_path, index=False)
        return
        
        