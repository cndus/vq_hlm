#!/bin/bash

#SBATCH --account=yfliu3
#SBATCH --job-name=residualvq
#SBATCH --partition=RTX3090,RTX4090,A100 # 用sinfo命令可以看到所有队列
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡或多进程，请调整此参数
#SBATCH --cpus-per-task=16  # 每个进程的CPU数量
#SBATCH --gres=gpu:1        # 若使用2块卡，则gres=gpu:2
#SBATCH --output=./runs/residualvq/%j.out
#SBATCH --error=./runs/residualvq/%j.err
#SBATCH --mail-type=all     # 设置邮件通知类型，可选all, end, fail, begin
#SBATCH --mail-user=1729372667@qq.com # 设置通知邮箱

python train_vq.py --ckpt_dir ./runs/residualvq --model_config conf/models/residualvq.yaml
python train_vq.py --ckpt_dir ./runs/residualvq --model_config conf/models/residualvq.yaml --test