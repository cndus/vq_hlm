#!/bin/bash

#SBATCH --account=xhyin
#SBATCH --job-name=rSimvq
#SBATCH --partition=A100 # 用sinfo命令可以看到所有队列 RTX4090
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --cpus-per-task=16  # 每个进程的CPU数量
#SBATCH --gres=gpu:1        # 若使用2块卡，则gres=gpu:2


python train_vq.py --ckpt_dir runs/residualSimvq --model_config conf/models/residualSimvq.yaml
