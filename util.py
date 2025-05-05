import os
import torch
import random
import numpy as np
import shutil

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
    # 保存最新检查点
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, os.path.join(checkpoint_dir, filename))
    # 如果是最佳模型，复制一份
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        # 只有当路径不同时才复制文件
        if checkpoint_path != best_path:
            shutil.copyfile(checkpoint_path, best_path)
        
def load_checkpoint(checkpoint_dir, filename='checkpoint.pt'):
    """加载检查点"""
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    if os.path.exists(checkpoint_path):
        return torch.load(checkpoint_path)
    return None

def calculate_bleu(references, hypotheses):
    """计算BLEU分数"""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    # 确保参考翻译列表的格式为[[tokens]]
    formatted_refs = [[ref.split()] for ref in references]
    # 确保假设翻译列表的格式为[tokens]
    formatted_hyps = [hyp.split() for hyp in hypotheses]
    
    smoothie = SmoothingFunction().method1
    return corpus_bleu(
        formatted_refs, 
        formatted_hyps, 
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie
    )
    
def write_translations(translations, output_file):
    """将翻译结果写入文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for translation in translations:
            f.write(translation + '\n')
            
def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device():
    """获取可用设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
