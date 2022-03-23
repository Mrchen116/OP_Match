CUDA_VISIBLE_DEVICES=$2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29116 main.py --dataset imagenet --out $1 --arch resnet_imagenet --lambda_oem 0.1 --lambda_socr 0.5 \
--batch-size 128 --lr 0.03 --expand-labels --seed 0 --opt_level O2 --amp --mu 2 --epochs 100 --num-workers 8







