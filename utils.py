import yaml

# 读取 YAML 配置文件
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def count_parameters(model):
    # 统计可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 统计不可训练参数量
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    return trainable_params, non_trainable_params