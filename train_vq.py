import argparse
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from constants import KEY_LM_HIDDEN_STATES
from dataloading import get_chunked_h5dataloader
from vector_quantize_pytorch import VectorQuantize
import logging
import os
from models import get_model


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

lr = 3e-4
train_epochs = 1
num_codes = 1024
seed = 1234
device = "cuda" if torch.cuda.is_available() else "cpu"


def save_checkpoint(model, optimizer, step, ckpt_path):
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckpt_path)

def load_checkpoint(model, optimizer, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step']

def compute_loss(model, x, alpha=10):
    out, indices, cmt_loss = model(x)
    out = out.clamp(-1., 1.)
    rec_loss = (out - x).abs().mean()
    total_loss = rec_loss + alpha * cmt_loss
    return rec_loss, cmt_loss, total_loss, indices

def evaluate(model, eval_loader, split:str, writer:SummaryWriter=None, step:int=None):
    model.to(device)
    model.eval()
    eval_loss = 0
    index_count = {i: 0 for i in range(num_codes)}
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Running on {split}"):
            x = batch[KEY_LM_HIDDEN_STATES].to(device)
            rec_loss, cmt_loss, total_loss, indices = compute_loss(model, x)
            eval_loss += total_loss.item()
            unique_indices = indices.unique().cpu().numpy()
            for idx in unique_indices:
                index_count[idx] += 1

    eval_loss /= len(eval_loader)
    utilized_indices = sum(1 for count in index_count.values() if count > 0)
    active_percent = utilized_indices / num_codes * 100
    logging.info(f"{split} Loss: {eval_loss:.4f}")
    logging.info(f'{split} Active Percentage: {active_percent:.4f}')
    if writer:
        writer.add_scalar(f'Loss/{split}', eval_loss, step)
        writer.add_scalar(f'Active/{split}', active_percent, step)
    return eval_loss, index_count

def save_histogram(args, index_count):
    import csv
    import matplotlib.pyplot as plt
    with open(os.path.join(args.ckpt_dir, 'index_frequencies.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'Frequency'])  # 写入表头
        for idx, count in index_count.items():
            writer.writerow([idx, count])  # 写入每个索引的频次

    logging.info("Index frequencies saved to 'index_frequencies.csv'")
    
    frequencies = list(index_count.values())

    plt.bar(range(num_codes), frequencies, edgecolor='black', alpha=0.7)
    plt.title("Frequency of Codebook Indices (Entire Dataset)")
    plt.xlabel("Codebook Index")
    plt.ylabel("Frequency")
    plt.xticks(range(0, num_codes, 50))
    plt.savefig(os.path.join(args.ckpt_dir, 'index_frequencies.png'))


def train(model, args, train_loader, val_loader=None, train_epochs=1, alpha=10, validate_every=1000, writer=None, resume_from_step=0):
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    step = resume_from_step

    # Load checkpoint if resuming
    if resume_from_step > 0:
        ckpt_path = os.path.join(args.ckpt_dir, 'latest_checkpoint.pt')
        step = load_checkpoint(model, opt, ckpt_path)
        logging.info(f"Resumed from step {resume_from_step}")
    
    for epoch in range(train_epochs):
        logging.info(f"Starting epoch {epoch}")
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            opt.zero_grad()
            x = batch[KEY_LM_HIDDEN_STATES].to(device)
            rec_loss, cmt_loss, total_loss, indices = compute_loss(model, x, alpha)
            total_loss.backward()

            opt.step()
            active_percent = indices.unique().numel() / num_codes * 100
            pbar.set_description(
                f"rec loss: {rec_loss.item():.3f} | "
                + f"cmt loss: {cmt_loss.item():.3f} | "
                + f"active %: {active_percent:.3f}"
            )

            if writer:
                writer.add_scalar('Loss/Train', total_loss.item(), step)
                writer.add_scalar('Active/Train', active_percent, step)
            step += 1

            if val_loader and step % validate_every == 0:
                val_loss, _ = evaluate(model, val_loader, "Validation", writer, step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, opt, step, os.path.join(args.ckpt_dir, 'best_checkpoint.pt'))

        save_checkpoint(model, opt, step, os.path.join(args.ckpt_dir, 'latest_checkpoint.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", default='conf/data/example.yaml')
    parser.add_argument("--model_config", default='conf/models/vectorquantize.yaml')
    parser.add_argument("--ckpt_dir", default='./checkpoints')
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    train_dataloader = get_chunked_h5dataloader(config_path=args.data_config, split='train')
    val_dataloader = get_chunked_h5dataloader(config_path=args.data_config, split='validation')
    test_dataloader = get_chunked_h5dataloader(config_path=args.data_config, split='test')

    torch.manual_seed(seed)
    model = get_model(args.model_config)

    writer = SummaryWriter(log_dir=os.path.join(args.ckpt_dir, 'logs'))
    if not args.test:
        train(model, args, train_dataloader, val_dataloader, train_epochs=train_epochs, writer=writer)

    # Test using best checkpoint
    logging.info("Loading best checkpoint for testing")
    load_checkpoint(model, None, os.path.join(args.ckpt_dir, 'best_checkpoint.pt'))
    _, index_count = evaluate(model, test_dataloader, "Test", writer)
    save_histogram(args, index_count)

    writer.close()
