#!/bin/bash

#SBATCH --account=xhyin
#SBATCH --job-name=Simvq
#SBATCH --partition=RTX3090,RTX4090,A100 # 用sinfo命令可以看到所有队列
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --cpus-per-task=16  # 每个进程的CPU数量
#SBATCH --gres=gpu:2        # 若使用2块卡，则gres=gpu:2


python train_vq.py --ckpt_dir ./runs/Simvq --model_config conf/models/Simvq.yaml
python train_vq.py --ckpt_dir ./runs/Simvq --model_config conf/models/Simvq.yaml --test