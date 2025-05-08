import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import os
import random
import pickle

class Vocabulary:
    def __init__(self, pad_token="<pad>", unk_token="<unk>", 
                 bos_token="<bos>", eos_token="<eos>"):
        self.token2idx = {
            pad_token: 0,
            unk_token: 1,
            bos_token: 2,
            eos_token: 3
        }
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
    def build_vocab(self, sentences, min_freq=5):
        word_count = {}
        
        # 统计词频
        for sentence in sentences:
            for word in sentence.split():
                word_count[word] = word_count.get(word, 0) + 1
        
        # 按词频添加到词汇表
        idx = len(self.token2idx)
        for word, count in word_count.items():
            if count >= min_freq:
                self.token2idx[word] = idx
                self.idx2token[idx] = word
                idx += 1
                
    def __len__(self):
        return len(self.token2idx)
        
    def encode(self, sentence, add_special_tokens=True):
        """将句子转换为索引序列"""
        if isinstance(sentence, str):
            tokens = sentence.split()
        else:
            tokens = sentence
            
        ids = [self.token2idx.get(token, self.unk_token_id) for token in tokens]
        
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
            
        return ids
        
    def decode(self, ids, skip_special_tokens=True):
        """将索引序列转换回句子"""
        tokens = []
        for idx in ids:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            token = self.idx2token.get(idx, "<unk>")
            if skip_special_tokens and token in ["<pad>", "<bos>", "<eos>"]:
                continue
            tokens.append(token)
            
        return " ".join(tokens)
    
    def save(self, path):
        """保存词汇表到文件"""
        torch.save({
            'token2idx': self.token2idx,
            'idx2token': self.idx2token,
            'pad_token_id': self.pad_token_id,
            'unk_token_id': self.unk_token_id,
            'bos_token_id': self.bos_token_id,
            'eos_token_id': self.eos_token_id
        }, path)
    
    @classmethod
    def load(cls, path):
        """从文件加载词汇表"""
        data = torch.load(path)
        vocab = cls()
        vocab.token2idx = data['token2idx']
        vocab.idx2token = data['idx2token']
        vocab.pad_token_id = data['pad_token_id']
        vocab.unk_token_id = data['unk_token_id']
        vocab.bos_token_id = data['bos_token_id']
        vocab.eos_token_id = data['eos_token_id']
        return vocab

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences=None, max_len=100):
        """
        初始化翻译数据集
        
        Args:
            src_sentences: 源语言句子列表
            tgt_sentences: 目标语言句子列表，如果为None，则为测试集
            max_len: 最大序列长度
        """
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.max_len = max_len
        self.is_test = tgt_sentences is None
        
        if not self.is_test:
            assert len(self.src_sentences) == len(self.tgt_sentences), "源语言和目标语言句子数量不匹配"
            
    def __len__(self):
        return len(self.src_sentences)
        
    def __getitem__(self, idx):
        # 截断到最大长度
        src_tokens = self.src_sentences[idx].split()[:self.max_len]
        
        if self.is_test:
            return {
                'src': src_tokens,
                'src_text': ' '.join(src_tokens),
            }
        else:
            tgt_tokens = self.tgt_sentences[idx].split()[:self.max_len]
            return {
                'src': src_tokens,
                'tgt': tgt_tokens,
                'src_text': ' '.join(src_tokens),
                'tgt_text': ' '.join(tgt_tokens)
            }

def save_processed_data(data, file_path):
    """保存处理过的数据到磁盘"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"数据已保存到: {file_path}")

def load_processed_data(file_path):
    """从磁盘加载处理过的数据"""
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"数据已从 {file_path} 加载")
    return data

def load_and_split_data(src_file, tgt_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, 
                        seed=42, data_dir="processed_data"):
    """
    加载并分割数据为训练集、验证集和测试集，支持保存和加载处理过的数据
    
    Args:
        src_file: 源语言文件路径
        tgt_file: 目标语言文件路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        data_dir: 处理后数据存储目录
    
    Returns:
        train_src, train_tgt, val_src, val_tgt, test_src, test_tgt
    """
    # 确保比例之和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    # 创建数据存储目录
    os.makedirs(data_dir, exist_ok=True)
    
    # 构建保存文件路径
    data_path = os.path.join(data_dir, f"split_data_tr{train_ratio}_vl{val_ratio}_ts{test_ratio}_s{seed}.pkl")
    
    # 尝试加载已处理的数据
    loaded_data = load_processed_data(data_path)
    if loaded_data:
        train_src, train_tgt, val_src, val_tgt, test_src, test_tgt = loaded_data
        print(f"数据集已加载，总计 {len(train_src) + len(val_src) + len(test_src)} 条数据")
        print(f"训练集: {len(train_src)} 条，验证集: {len(val_src)} 条，测试集: {len(test_src)} 条")
        return train_src, train_tgt, val_src, val_tgt, test_src, test_tgt
    
    # 读取源语言和目标语言文件
    with open(src_file, 'r', encoding='utf-8') as f:
        src_sentences = [line.strip() for line in f]
    
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_sentences = [line.strip() for line in f]
    
    # 确保数量一致
    assert len(src_sentences) == len(tgt_sentences), "源语言和目标语言句子数量不匹配"
    
    # 设置随机种子
    random.seed(seed)
    
    # 打乱并保持一致性
    combined = list(zip(src_sentences, tgt_sentences))
    random.shuffle(combined)
    src_sentences, tgt_sentences = zip(*combined)
    
    # 计算数据集大小
    total_size = len(src_sentences)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # 分割数据集
    train_src = src_sentences[:train_size]
    train_tgt = tgt_sentences[:train_size]
    
    val_src = src_sentences[train_size:train_size+val_size]
    val_tgt = tgt_sentences[train_size:train_size+val_size]
    
    test_src = src_sentences[train_size+val_size:]
    test_tgt = tgt_sentences[train_size+val_size:]
    
    print(f"数据集分割完成，总计 {total_size} 条数据")
    print(f"训练集: {len(train_src)} 条，验证集: {len(val_src)} 条，测试集: {len(test_src)} 条")
    
    # 保存处理过的数据
    data_to_save = (train_src, train_tgt, val_src, val_tgt, test_src, test_tgt)
    save_processed_data(data_to_save, data_path)
    
    return train_src, train_tgt, val_src, val_tgt, test_src, test_tgt

def collate_fn(batch, src_vocab, tgt_vocab=None):
    """
    数据批处理函数，将批次数据填充到相同长度
    """
    is_test = tgt_vocab is None
    
    src_tokens = [item['src'] for item in batch]
    src_encodings = [torch.tensor(src_vocab.encode(tokens), dtype=torch.long) for tokens in src_tokens]
    src_padded = rnn_utils.pad_sequence(src_encodings, batch_first=True, padding_value=src_vocab.pad_token_id)
    
    result = {
        'src': src_padded,
        'src_text': [item['src_text'] for item in batch],
    }
    
    if not is_test:
        tgt_tokens = [item['tgt'] for item in batch]
        tgt_encodings = [torch.tensor(tgt_vocab.encode(tokens), dtype=torch.long) for tokens in tgt_tokens]
        tgt_padded = rnn_utils.pad_sequence(tgt_encodings, batch_first=True, padding_value=tgt_vocab.pad_token_id)
        
        result.update({
            'tgt': tgt_padded,
            'tgt_text': [item['tgt_text'] for item in batch],
        })
    
    return result

def get_dataloader(dataset, batch_size, src_vocab, tgt_vocab=None, shuffle=False):
    """获取数据加载器"""
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=lambda batch: collate_fn(batch, src_vocab, tgt_vocab)
    )

def build_vocabs(train_src, train_tgt, min_freq=5):
    """构建源语言和目标语言的词汇表"""
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    
    # 构建源语言词汇表
    src_vocab.build_vocab(train_src, min_freq)
    
    # 构建目标语言词汇表
    tgt_vocab.build_vocab(train_tgt, min_freq)
    
    return src_vocab, tgt_vocab