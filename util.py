import os
import torch
import random
import numpy as np
import shutil
import json
import pickle

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

def save_dataset(train_data, val_data, test_data, save_dir='data'):
    """
    保存训练、验证和测试数据集
    
    Args:
        train_data: 训练数据集 (src, tgt)的元组
        val_data: 验证数据集 (src, tgt)的元组
        test_data: 测试数据集 (src, tgt)的元组
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    train_src, train_tgt = train_data
    val_src, val_tgt = val_data
    test_src, test_tgt = test_data
    
    # 保存数据集
    with open(os.path.join(save_dir, 'train_src.pkl'), 'wb') as f:
        pickle.dump(train_src, f)
    with open(os.path.join(save_dir, 'train_tgt.pkl'), 'wb') as f:
        pickle.dump(train_tgt, f)
    
    with open(os.path.join(save_dir, 'val_src.pkl'), 'wb') as f:
        pickle.dump(val_src, f)
    with open(os.path.join(save_dir, 'val_tgt.pkl'), 'wb') as f:
        pickle.dump(val_tgt, f)
    
    with open(os.path.join(save_dir, 'test_src.pkl'), 'wb') as f:
        pickle.dump(test_src, f)
    with open(os.path.join(save_dir, 'test_tgt.pkl'), 'wb') as f:
        pickle.dump(test_tgt, f)
    
    # 保存数据集信息
    dataset_info = {
        'train_size': len(train_src),
        'val_size': len(val_src),
        'test_size': len(test_src)
    }
    
    with open(os.path.join(save_dir, 'dataset_info.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    print(f"数据集已保存到 {save_dir} 目录")
    print(f"训练集: {dataset_info['train_size']} 条")
    print(f"验证集: {dataset_info['val_size']} 条")
    print(f"测试集: {dataset_info['test_size']} 条")

def load_dataset(save_dir='data'):
    """
    加载保存的数据集
    
    Args:
        save_dir: 数据集保存目录
        
    Returns:
        train_data, val_data, test_data: 加载的数据集，每个数据集为(src, tgt)元组
    """
    # 加载数据集
    with open(os.path.join(save_dir, 'train_src.pkl'), 'rb') as f:
        train_src = pickle.load(f)
    with open(os.path.join(save_dir, 'train_tgt.pkl'), 'rb') as f:
        train_tgt = pickle.load(f)
    
    with open(os.path.join(save_dir, 'val_src.pkl'), 'rb') as f:
        val_src = pickle.load(f)
    with open(os.path.join(save_dir, 'val_tgt.pkl'), 'rb') as f:
        val_tgt = pickle.load(f)
    
    with open(os.path.join(save_dir, 'test_src.pkl'), 'rb') as f:
        test_src = pickle.load(f)
    with open(os.path.join(save_dir, 'test_tgt.pkl'), 'rb') as f:
        test_tgt = pickle.load(f)
    
    # 加载数据集信息
    with open(os.path.join(save_dir, 'dataset_info.json'), 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    print(f"已加载数据集:")
    print(f"训练集: {dataset_info['train_size']} 条")
    print(f"验证集: {dataset_info['val_size']} 条")
    print(f"测试集: {dataset_info['test_size']} 条")
    
    return (train_src, train_tgt), (val_src, val_tgt), (test_src, test_tgt)
