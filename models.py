from vector_quantize_pytorch import VectorQuantize, ResidualVQ, GroupedResidualVQ, RandomProjectionQuantizer, SimVQ, ResidualSimVQ, LFQ
import torch.nn as nn
from vector_quantize_pytorch import Sequential
from custom_models.truthx import TruthXVAE
from utils import load_config, count_parameters

import yaml
import torch
import torch.nn.functional as F

FIRST_PROJECT_DIM = 2048
SECOND_PROJECT_DIM = 1024

# 读取 YAML 配置文件
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params

def get_model(vae_config):
    if vae_config['vq_type'] == 'VectorQuantize': 
        vqvae = VectorQuantize(
            dim=vae_config['embedding_dim'],
            codebook_dim=vae_config['codebook_dim'],
            codebook_size=vae_config['codebook_size'],
            decay=0.8,
            commitment_weight=1.,
            kmeans_init=True,
            rotation_trick=True,
            straight_through=False,
        )
    elif vae_config['vq_type'] == 'ResidualVQ':
        vqvae = ResidualVQ(
            dim = vae_config['embedding_dim'],
            codebook_dim=vae_config['codebook_dim'],
            num_quantizers = vae_config['num_quantizers'],      # specify number of quantizers
            codebook_size = vae_config['codebook_size'],    # codebook size
        )
    elif vae_config['vq_type'] == 'TruthX_ResidualVQ':
        vqvae = TruthXVAE(vae_config)
    elif vae_config['vq_type'] == 'GroupedResidualVQ':  
        vqvae = GroupedResidualVQ(
            dim = vae_config['embedding_dim'],
            num_quantizers = vae_config['num_quantizers'] // vae_config['num_groups'],      # specify number of quantizers
            groups = vae_config['num_groups'],
            codebook_size = vae_config['codebook_size'],    # codebook size
        )
    elif vae_config['vq_type'] == 'RandomProjectionQuantizer':  
        vqvae = RandomProjectionQuantizer(
            dim = vae_config['embedding_dim'],               # input dimensions
            num_codebooks = vae_config['num_quantizers'],      # in USM, they used up to 16 for 5% gain
            codebook_dim = vae_config['codebook_dim'],      # codebook dimension
            codebook_size = vae_config['codebook_size']     # codebook size
        )
    elif vae_config['vq_type'] == 'SimVQ': 
        vqvae = SimVQ(
            dim = vae_config['embedding_dim'],
            codebook_size = vae_config['codebook_size'],
            rotation_trick = True  # use rotation trick from Fifty et al.
        )
    elif vae_config['vq_type'] == 'ResidualSimVQ': 
        vqvae = ResidualSimVQ(
            dim = vae_config['embedding_dim'],
            num_quantizers = vae_config['num_quantizers'],
            codebook_size = vae_config['codebook_size'],
            rotation_trick = True  # use rotation trick from Fifty et al.
        )
    elif vae_config['vq_type'] == 'LFQ': 
        vqvae = LFQ(
            codebook_size = vae_config['codebook_size'],      # codebook size, must be a power of 2
            dim = vae_config['embedding_dim'],                   # this is the input feature dimension, defaults to log2(codebook_size) if not defined
            entropy_loss_weight = 0.1,  # how much weight to place on entropy loss
            diversity_gamma = 1.        # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
        )

    n_trainable, n_fixed = count_parameters(vqvae)
    print(f'Model parameter stats for {vae_config["vq_type"]}')
    print(f'{n_trainable / 1e6:.2f}M')
    return vqvae