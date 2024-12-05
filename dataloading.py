import os
import h5py
from torch.utils.data import Dataset, DataLoader
import torch
from utils import load_config


class HDF5Dataset(Dataset):
    def __init__(self, h5_files_dir, split):
        """
        Args:
            h5_files_dir (str): 
            split (str): train/validation/test
        """
        h5_file_path = os.path.join(h5_files_dir, split+'.h5')
        self.h5_file = h5py.File(h5_file_path, 'r')
        
        # 获取数据集
        self.hidden_states = self.h5_file['hidden_states']
        self.input_ids = self.h5_file['input_ids']
        self.labels = self.h5_file['labels']
        
        # 获取总样本数
        self.total_samples = self.hidden_states.shape[0]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        hidden_states = torch.tensor(self.hidden_states[idx], dtype=torch.float)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'hidden_states': hidden_states
        }

    def close(self):
        self.h5_file.close()


class ChunkedHDF5Dataset(HDF5Dataset):
    def __init__(self, h5_files_dir, split, chunk_size:int):
        super().__init__(h5_files_dir, split)
        self.chunk_size = chunk_size
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        hidden_states = item['hidden_states']
        item['hidden_states'] = hidden_states[self.chunk_size-1::self.chunk_size, :]
        return item


if __name__ == '__main__':
    config = load_config(config_path='conf/example.yaml')
    split = 'test'
    dataset = ChunkedHDF5Dataset(config['h5_file_path'], split, chunk_size=config['chunk_size'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    for batch in dataloader:
        input_ids = batch['input_ids']
        labels = batch['labels']
        hidden_states = batch['hidden_states']
        
        print(f"Input IDs: {input_ids.shape}")
        print(f"Labels: {labels.shape}")
        print(f"Hidden States: {hidden_states.shape}")
        break  # 这里只打印一个批次的数据
