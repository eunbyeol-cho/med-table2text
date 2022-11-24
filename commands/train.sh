device_num=$1
ckpt_name=$2
mlm_prob=$3

OMP_NUM_THREADS=8 \
NUMEXPR_MAX_THREADS=128 \
CUDA_VISIBLE_DEVICES=$device_num \
python3 ../main.py \
--ckpt_name=$ckpt_name \
--batch_size=16 \
--sub_root=znorm \
--mask_null \
--mlm_prob=$mlm_prob \
--date_root=22-07 \
--patience=20 \
--lr=5e-5 \
--device_num=$device_num \
--wandb_project_name={wandb_project_name} \
--model=both2text