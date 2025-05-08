import os
import torch
import random
import numpy as np
import shutil
import sacrebleu

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pt'):
    """保存检查点"""
    # 确保目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 保存最新检查点
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    
    # 如果是最佳模型，复制一份
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        # 只有当路径不同时才复制文件
        if checkpoint_path != best_path:
            shutil.copyfile(checkpoint_path, best_path)
        
def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device=None):
    """加载检查点"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device if device else 'cpu')
        
        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # 处理多GPU训练保存的模型
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            print(f"Model loaded from {checkpoint_path}")
        else:
            print(f"Warning: model_state_dict not found in checkpoint")
        
        # 加载优化器权重
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Optimizer loaded from {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Failed to load optimizer state: {e}")
        
        # 加载调度器权重
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"Scheduler loaded from {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Failed to load scheduler state: {e}")
        
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def calculate_bleu(references, hypotheses):
    """使用sacrebleu计算BLEU分数"""
    return sacrebleu.corpus_bleu(hypotheses, [references], tokenize='13a', lowercase=True).score / 100.0
    
def write_translations(src_texts, ref_texts, hyp_texts, output_file):
    """将翻译结果写入文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for src, ref, hyp in zip(src_texts, ref_texts, hyp_texts):
            f.write(f"SRC: {src}\n")
            f.write(f"REF: {ref}\n")
            f.write(f"HYP: {hyp}\n\n")
            
def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # 对于MacOS上的Apple Silicon
    else:
        return torch.device("cpu")
