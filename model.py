import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    """Transformer模型"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, 
                 dropout=0.1, activation="relu"):
        super(TransformerModel, self).__init__()
        
        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        
        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化参数
        self._reset_parameters()
        
        # 存储模型参数
        self.d_model = d_model
        self.nhead = nhead
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
    def _reset_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, 
                src_padding_mask=None, tgt_padding_mask=None, 
                memory_mask=None, memory_key_padding_mask=None):
        """前向传播"""
        # 创建源语言padding mask
        if src_padding_mask is None:
            src_padding_mask = (src == 0)
            
        # 编码器前向传播
        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
        
        if tgt is None:  # 用于推理阶段
            return memory
            
        # 创建目标语言subsequent mask
        if tgt_mask is None:
            tgt_len = tgt.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)
            
        # 创建目标语言padding mask
        if tgt_padding_mask is None:
            tgt_padding_mask = (tgt == 0)
            
        # 解码器前向传播
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        output = self.transformer.decoder(
            tgt_emb, memory, 
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # 映射到目标词汇表大小
        output = self.output_layer(output)
        
        return output
        
    def greedy_decode(self, src, src_padding_mask=None, max_len=100, bos_id=2, eos_id=3):
        """贪婪解码，用于推理阶段"""
        batch_size = src.size(0)
        device = src.device
        
        # 编码源语言
        memory = self.forward(src, src_padding_mask=src_padding_mask)
        
        # 初始化目标序列
        ys = torch.ones(batch_size, 1).fill_(bos_id).long().to(device)
        
        for i in range(max_len - 1):
            # 解码当前序列
            out = self.forward(
                src, ys, 
                src_padding_mask=src_padding_mask,
                tgt_mask=None,  # 会自动创建
                tgt_padding_mask=None,  # 会自动创建
                memory_key_padding_mask=src_padding_mask
            )
            
            # 获取最后一个时间步的预测
            prob = out[:, -1]
            next_word = torch.argmax(prob, dim=-1).unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)
            
            # 检查是否所有句子都生成了结束标记
            if ((next_word == eos_id).sum() == batch_size):
                break
                
        return ys
        
    def beam_search(self, src, src_padding_mask=None, max_len=100, beam_size=5, bos_id=2, eos_id=3):
        """集束搜索，用于更高质量的推理"""
        batch_size = src.size(0)
        device = src.device
        
        # 编码源语言
        memory = self.forward(src, src_padding_mask=src_padding_mask)
        
        # 对每个样本单独进行集束搜索
        all_hypotheses = []
        
        for i in range(batch_size):
            # 获取当前样本的编码
            sample_memory = memory[i:i+1]  # [1, src_len, d_model]
            sample_padding_mask = src_padding_mask[i:i+1] if src_padding_mask is not None else None
            
            # 初始候选序列
            hypotheses = [([bos_id], 0.0)]
            completed_hypotheses = []
            
            for _ in range(max_len - 1):
                new_hypotheses = []
                
                for seq, score in hypotheses:
                    if seq[-1] == eos_id:
                        completed_hypotheses.append((seq, score))
                        continue
                        
                    # 构建当前序列tensor
                    curr_seq = torch.tensor([seq], dtype=torch.long).to(device)
                    
                    # 解码当前序列
                    out = self.forward(
                        src[i:i+1], curr_seq,
                        src_padding_mask=src_padding_mask[i:i+1] if src_padding_mask is not None else None,
                        memory_key_padding_mask=sample_padding_mask
                    )
                    
                    # 获取最后一个时间步的预测
                    logits = out[0, -1]
                    probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    
                    # 获取topk
                    topk_probs, topk_indices = probs.topk(beam_size)
                    
                    for prob, idx in zip(topk_probs, topk_indices):
                        new_seq = seq + [idx.item()]
                        new_score = score + prob.item()
                        new_hypotheses.append((new_seq, new_score))
                
                # 只保留top-beam_size个候选
                hypotheses = sorted(new_hypotheses, key=lambda x: x[1], reverse=True)[:beam_size]
                
                # 如果所有候选都已完成，则停止
                if all(h[0][-1] == eos_id for h in hypotheses):
                    completed_hypotheses.extend(hypotheses)
                    break
            
            # 如果没有完成的候选，则添加当前的候选
            if len(completed_hypotheses) == 0:
                completed_hypotheses = hypotheses
                
            # 选择最佳候选
            best_hypothesis = max(completed_hypotheses, key=lambda x: x[1])
            all_hypotheses.append(torch.tensor(best_hypothesis[0], dtype=torch.long).to(device))
            
        # 将所有候选填充到相同长度
        max_hyp_len = max(len(h) for h in all_hypotheses)
        result = torch.zeros(batch_size, max_hyp_len, dtype=torch.long).to(device)
        
        for i, hyp in enumerate(all_hypotheses):
            result[i, :len(hyp)] = hyp
            
        return result
