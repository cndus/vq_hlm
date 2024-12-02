import h5py
import numpy as np

# 读取 HDF5 文件
with h5py.File('/data1/yfliu/vqhlm/datasets/wikitext103_gpt2finetuned/test.h5', 'r') as f:
    # 读取 total_samples 元数据
    total_samples = f.attrs['total_samples']
    print(f"Total samples in the dataset: {total_samples}")
    
    # 读取数据集
    hidden_states = f['hidden_states'][:]
    input_ids = f['input_ids'][:]
    labels = f['labels'][:]

    # 打印数据集的形状，确保读取正确
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # 获取第一个样本的数据并展示
    sample_idx = 0  # 举例，读取第一个样本
    print(f"Sample {sample_idx} - Input IDs: {input_ids[sample_idx]}")
    print(f"Sample {sample_idx} - Labels: {labels[sample_idx]}")
    
    # 获取对应的 hidden states (flattened)
    print(f"Sample {sample_idx} - Hidden States: {hidden_states[sample_idx]}")
