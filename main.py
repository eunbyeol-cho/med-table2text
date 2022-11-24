import os, glob, sys, random, pickle
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
import warnings
import logging
import logging.config

warnings.filterwarnings(action='ignore')

logging.basicConfig(
    format="%(asctime)s | %(levelname)s %(name)s %(message)s)))",
    datefmt="%Y-%m-%d %H:%M:%S",
    level = os.environ.get("LOGLEVEL", "INFO").upper(),
    stream = sys.stdout
)
logger = logging.getLogger(__name__)


def get_parser():
    
    parser = argparse.ArgumentParser()

    #Data args
    parser.add_argument('--root', type=str, default='/home/data_storage/mimic3/snuh')
    parser.add_argument('--sub_root', type=str, default='znorm', choices=['znorm', 'minmax'])
    parser.add_argument('--date_root', type=str, default='22-07')
    parser.add_argument('--valid_subsets', type=str, default="valid, test")
    parser.add_argument('--fixed_test_mask', type=str, default=None, choices=['valid', 'test', 'total', None], help="test on only testset or whole dataset")
    parser.add_argument('--text_vocab', type=int, default=119547, help='vocab size in bert-base-multilingual-cased tokneizer')

    #Model args
    parser.add_argument('--model', type=str, choices=['both2table', 'table2table', 'both2text', 'text2text'], default='both2table')
    parser.add_argument('--sep_regressor', action='store_true', help='separate numeric regressor') 
    parser.add_argument('--sep_numemb', action='store_true', help='separate numeric embedding') 
    parser.add_argument('--emb_dim', type=int, default=256)  

    #MLM related args
    parser.add_argument('--mlm_prob', type=float, default=0.3)
    parser.add_argument('--valid_mlm_prob', type=float, default=None,required=False)
    parser.add_argument('--mask_null', action='store_true', help='masking null tokens as well')
    parser.add_argument('--var_abs', type=str, choices=['abs', 'square', 'relu'], default='abs')
    parser.add_argument('--condition_aki', action='store_true', help='condition aki=0/1')

    #Training args
    parser.add_argument('--device_num', type=str, default='2,3')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--fold', type=str, default=None)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--unnorm', action='store_true', help='Calculate unnormalzied mse')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--ignore_pad', action='store_true')
    parser.add_argument('--num_loss_weight', type=float, default=1)

    #Save
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--save_prefix', type=str, default='checkpoint')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--ckpt_name', required=True)

    #Wandb setting
    parser.add_argument('--wandb_project_name', type=str, default='CorrectMasking_3way&Include_Op_Code')
    parser.add_argument('--wandb_run_name', type=str)
    parser.add_argument('--debug', action='store_true')

    #Evaluation
    parser.add_argument('--analyze', action='store_true', help='graphical visualization & statistical test')
    parser.add_argument('--null_classifier', action='store_true')
    return parser


def add_sub_parser(parser):

    args = parser.parse_args()

    root = args.root
    sub_root = args.sub_root
    date_root = args.date_root
    mlm_prob = args.mlm_prob
    parser.add_argument('--input_path', type=str, default='{}/{}/{}'.format(root, date_root, sub_root))

    #Pre-defined args
    with open('{}/{}/{}/info_dict.pickle'.format(root, date_root, sub_root), 'rb') as fr:
        arg_dict = pickle.load(fr)
        # print(arg_dict)
        nc = arg_dict['cat_col_num']
        nn = arg_dict['num_col_num']
        nt = arg_dict['text_num']
        cat_null_id = arg_dict['cat_null_ids']
        num_null_id = arg_dict['num_null_ids']
        cat_vocab_size = arg_dict['cat_vocab_size']+2
        column_order = arg_dict['column_order']
        cat_columns_numclass_list = arg_dict['cat_columns_numclass_list']
        
    parser.add_argument('--nc', type=int, default=nc, help='number of categorical columns in the table')
    parser.add_argument('--nn', type=int, default=nn, help='number of numerical columns in the table')
    parser.add_argument('--nt', type=int, default=nt, help='number of text columns in the table')
    parser.add_argument('--cat_null_id', type=int, default=cat_null_id, help='null_id of categoricyal data')
    parser.add_argument('--num_null_id', type=int, default=num_null_id, help='null_id of numeric data')
    parser.add_argument('--cat_vocab_size', type=int, default=cat_vocab_size)
    parser.add_argument('--column_order', type=list, default=column_order[:])
    parser.add_argument('--cat_columns_numclass_list', type=dict, default=cat_columns_numclass_list, help='list of # uniuqe elems of cat columns')

    return parser


def set_struct(cfg: dict):

    from datetime import datetime
    from pytz import timezone

    now = datetime.now()
    now = now.astimezone(timezone('Asia/Seoul'))      

    root = os.path.abspath(
        os.path.dirname(__file__)
        )
       
    ckpt_root = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(cfg["seed"], cfg['sub_root'], cfg['mlm_prob'], cfg['ckpt_name'], cfg['model'], cfg['unnorm'], cfg['sep_regressor'], cfg['sep_numemb'], cfg['lr'])

    if cfg['resume'] is not None:
        date = cfg['resume']
    else :
        date = now.strftime("%Y-%m-%d")

    output_dir = os.path.join(root, "outputs", date , ckpt_root)
    resume_num = len(glob.glob(os.path.join(output_dir,'*.log')))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    os.chdir(output_dir)
    
    job_logging_cfg = {
        'version': 1,
        'formatters': {
            'simple': {
                'format': '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler', 'formatter': 'simple', 'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler', 'formatter': 'simple', 'filename': 'train_{}.log'.format(resume_num)
            }
        },
        'root': {
            'level': 'INFO', 'handlers': ['console', 'file']
            },
        'disable_existing_loggers': False
    }
    logging.config.dictConfig(job_logging_cfg)

    cfg_dir = ".config"

    if not os.path.exists(cfg_dir):
        os.mkdir(cfg_dir)
        os.mkdir(cfg['save_dir'])

        with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:
            for k, v in cfg.items():
                print("{}: {}".format(k, v), file=f)
    else : 
        with open(os.path.join(cfg_dir, "config.yaml"), "a") as f:
            for k, v in cfg.items():
                print("{}: {}".format(k, v), file=f)

    return ckpt_root


def main():

    parser = add_sub_parser(get_parser())
    args = parser.parse_args()
    
    args.valid_subsets = args.valid_subsets.replace(' ','').split(',')

    # Sanity Check
    if (args.wandb_project_name is None) and (args.debug==False):
        raise AssertionError('wandb project name should not be null')
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_num)
    args.device_ids = list(range(len(args.device_num.split(','))))
    print('device_number : ', args.device_ids)
    ckpt_root = set_struct(vars(args))
    if args.wandb_run_name is None:
        args.wandb_run_name = ckpt_root
    
    #Seed pivotting
    mp.set_sharing_strategy('file_system')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True

    from trainers.text_trainer import Trainer
    trainer = Trainer(args)

    if args.eval:
        trainer.generate()
    else : 
        trainer.train()


if __name__ == '__main__':
    main()
    


