import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import os

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
    def __init__(self, src_file, tgt_file=None, max_len=100):
        """
        初始化翻译数据集
        
        Args:
            src_file: 源语言文件路径
            tgt_file: 目标语言文件路径，如果为None，则为测试集
            max_len: 最大序列长度
        """
        self.src_sentences = self.read_file(src_file)
        self.tgt_sentences = self.read_file(tgt_file) if tgt_file else None
        self.max_len = max_len
        self.is_test = tgt_file is None
        
        if not self.is_test:
            assert len(self.src_sentences) == len(self.tgt_sentences), "源语言和目标语言句子数量不匹配"
            
    def read_file(self, file_path):
        """读取文本文件"""
        if file_path is None:
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
            
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

def build_vocabs(train_dataset, min_freq=5):
    """构建源语言和目标语言的词汇表"""
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    
    # 构建源语言词汇表
    src_vocab.build_vocab([item['src_text'] for item in train_dataset], min_freq)
    
    # 构建目标语言词汇表
    tgt_vocab.build_vocab([item['tgt_text'] for item in train_dataset], min_freq)
    
    return src_vocab, tgt_vocab
