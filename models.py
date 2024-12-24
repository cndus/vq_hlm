from vector_quantize_pytorch import VectorQuantize, ResidualVQ, GroupedResidualVQ, RandomProjectionQuantizer, SimVQ, ResidualSimVQ, LFQ
from utils import load_config
import torch.nn as nn
from vector_quantize_pytorch import Sequential

def AEVQ(VQmodel):
    return Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VQmodel,
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )

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
        vqvae = AEVQ(SimVQ(
            dim = vae_config['embedding_dim'],
            codebook_size = vae_config['codebook_size'],
            rotation_trick = True  # use rotation trick from Fifty et al.
        ))
    elif vae_config['vq_type'] == 'ResidualSimVQ': 
        vqvae = AEVQ(ResidualSimVQ(
            dim = vae_config['embedding_dim'],
            num_quantizers = vae_config['num_quantizers'],
            codebook_size = vae_config['codebook_size'],
            rotation_trick = True,  # use rotation trick from Fifty et al.
            codebook_transform = nn.Sequential(
                nn.Linear(192, 768),
                nn.ReLU(),
                nn.Linear(768, 192),
            )
        ))
    elif vae_config['vq_type'] == 'LFQ': 
        vqvae = LFQ(
            codebook_size = vae_config['codebook_size'],      # codebook size, must be a power of 2
            dim = vae_config['embedding_dim'],                   # this is the input feature dimension, defaults to log2(codebook_size) if not defined
            entropy_loss_weight = 0.1,  # how much weight to place on entropy loss
            diversity_gamma = 1.        # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
        )

    return vqvae