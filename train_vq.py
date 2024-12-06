# FashionMnist VQ experiment with various settings.
# From https://github.com/minyoungg/vqtorch/blob/main/examples/autoencoder.py

from tqdm import tqdm

import torch
from constants import KEY_LM_HIDDEN_STATES
from dataloading import get_chunked_h5dataloader
from vector_quantize_pytorch import VectorQuantize
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

lr = 3e-4
train_epochs = 1
num_codes = 1024
hidden_size = 768
seed = 1234
rotation_trick = True
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, train_loader, train_epochs=1, alpha=10):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(train_epochs):
        logging.info(f"{epoch=}")
        pbar = tqdm(train_loader)
        for batch in pbar:
            opt.zero_grad()
            x = batch[KEY_LM_HIDDEN_STATES].to(device)

            out, indices, cmt_loss = model(x)
            out = out.clamp(-1., 1.)

            rec_loss = (out - x).abs().mean()
            (rec_loss + alpha * cmt_loss).backward()

            opt.step()
            pbar.set_description(
                f"rec loss: {rec_loss.item():.3f} | "
                + f"cmt loss: {cmt_loss.item():.3f} | "
                + f"active %: {indices.unique().numel() / num_codes * 100:.3f}"
            )


if __name__ == '__main__':
    train_dataloader = get_chunked_h5dataloader(config_path='conf/data/example.yaml', split='train')

    torch.random.manual_seed(seed)
    model = VectorQuantize(
        dim=hidden_size,
        codebook_dim=32,
        codebook_size = num_codes,     # codebook size
        decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
        commitment_weight = 1.,   # the weight on the commitment loss
        kmeans_init = True,
        rotation_trick = True,
        straight_through = False,
    )

    train(model, train_dataloader, train_epochs=train_epochs)
