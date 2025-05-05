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
        
        # 创建自定义TransformerEncoder和TransformerDecoder，以解决mask类型不匹配问题
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_encoder_layers
        )
        
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layers, 
            num_layers=num_decoder_layers
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
        # 创建源语言padding mask，确保类型一致
        if src_padding_mask is None:
            src_padding_mask = (src == 0).to(src.device)
            
        # 编码器前向传播
        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)
        
        if tgt is None:  # 用于推理阶段
            return memory
            
        # 创建目标语言subsequent mask
        if tgt_mask is None:
            tgt_len = tgt.size(1)
            device = tgt.device
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)
            
        # 创建目标语言padding mask，确保类型一致
        if tgt_padding_mask is None:
            tgt_padding_mask = (tgt == 0).to(tgt.device)
            
        # 设置memory key padding mask与src padding mask一致
        if memory_key_padding_mask is None:
            memory_key_padding_mask = src_padding_mask
            
        # 解码器前向传播
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        output = self.transformer_decoder(
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
        """优化的贪婪解码，用于推理阶段"""
        batch_size = src.size(0)
        device = src.device
        
        # 创建src_padding_mask
        if src_padding_mask is None:
            src_padding_mask = (src == 0).to(device)
        
        # 编码源语言
        memory = self.forward(src, src_padding_mask=src_padding_mask)
        
        # 初始化目标序列
        ys = torch.ones(batch_size, 1).fill_(bos_id).long().to(device)
        
        # 跟踪每个句子是否已完成
        completed = torch.zeros(batch_size, dtype=torch.bool).to(device)
        
        for i in range(max_len - 1):
            # 解码当前序列
            out = self.forward(
                src, ys, 
                src_padding_mask=src_padding_mask,
                memory_key_padding_mask=src_padding_mask
            )
            
            # 获取最后一个时间步的预测
            prob = out[:, -1]
            next_word = torch.argmax(prob, dim=-1).unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)
            
            # 更新完成状态
            completed = completed | (next_word.squeeze(1) == eos_id)
            
            # 如果所有句子都完成，则停止
            if completed.all():
                break
                
        return ys
        
    def beam_search(self, src, src_padding_mask=None, max_len=100, beam_size=5, bos_id=2, eos_id=3):
        """优化的束搜索，使用更高效的向量化实现"""
        device = src.device
        batch_size = src.size(0)
        
        # 处理src_padding_mask
        if src_padding_mask is None:
            src_padding_mask = (src == 0).to(device)
        
        # 编码源语言
        memory = self.forward(src, src_padding_mask=src_padding_mask)
        
        # 对每个样本单独进行束搜索
        final_outputs = []
        
        # 一次处理一个批次，避免显存溢出
        for b in range(batch_size):
            # 获取当前样本的编码和掩码
            single_memory = memory[b:b+1]
            single_src_padding_mask = src_padding_mask[b:b+1] if src_padding_mask is not None else None
            
            # 初始化束
            k_prev_words = torch.tensor([[bos_id]], dtype=torch.long, device=device)
            seqs = k_prev_words
            scores = torch.zeros(1, device=device)
            
            # 完成的序列
            complete_seqs = []
            complete_seqs_scores = []
            
            step = 1
            
            while True:
                if step > max_len:  # 如果超过最大长度，停止解码
                    break
                    
                # 获取当前序列长度
                num_words = seqs.size(1)
                
                # 推理：前向传播解码器
                output = self.forward(
                    src[b:b+1].expand(seqs.size(0), -1), 
                    seqs,
                    src_padding_mask=single_src_padding_mask.expand(seqs.size(0), -1) if single_src_padding_mask is not None else None,
                    memory_key_padding_mask=single_src_padding_mask.expand(seqs.size(0), -1) if single_src_padding_mask is not None else None
                )
                
                # 只关注最后一个时间步
                output = output[:, -1, :]
                scores_per_step = torch.log_softmax(output, dim=-1)
                
                # 添加到之前的分数
                scores = scores.unsqueeze(1).expand(-1, scores_per_step.size(1))
                scores = scores + scores_per_step
                
                # 对于第一步，始终从第一个束开始
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(beam_size, dim=0)
                else:
                    # 展平
                    top_k_scores, top_k_words = scores.view(-1).topk(beam_size, dim=0)
                
                # 将束索引转换为vocabulary索引
                prev_word_inds = top_k_words // scores_per_step.size(1)
                next_word_inds = top_k_words % scores_per_step.size(1)
                
                # 添加新词并更新分数
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
                scores = top_k_scores
                
                # 检查哪些序列已完成
                incomplete_inds = []
                complete_inds = []
                for i, next_word in enumerate(next_word_inds):
                    if next_word == eos_id:
                        complete_inds.append(i)
                    else:
                        incomplete_inds.append(i)
                        
                # 设置完成的序列
                for i in complete_inds:
                    complete_seqs.append(seqs[i].tolist())
                    complete_seqs_scores.append(scores[i].item())
                    
                # 检查停止条件
                beam_size = len(incomplete_inds)
                if beam_size == 0:
                    break
                    
                # 更新追踪的序列和分数
                seqs = seqs[incomplete_inds]
                scores = scores[incomplete_inds]
                
                # 更新Memory的bath size与新的束大小匹配
                single_memory = single_memory.expand(beam_size, -1, -1)
                if single_src_padding_mask is not None:
                    single_src_padding_mask = single_src_padding_mask.expand(beam_size, -1)
                    
                step += 1
                
            # 如果没有完成的序列，则使用当前最好的未完成序列
            if len(complete_seqs) == 0:
                best_seq = seqs[scores.argmax().item()].tolist()
            else:
                # 否则使用得分最高的完成序列
                max_score_idx = complete_seqs_scores.index(max(complete_seqs_scores))
                best_seq = complete_seqs[max_score_idx]
                
            final_outputs.append(torch.tensor(best_seq, device=device))
            
        # 填充所有序列到相同长度
        max_len = max(len(seq) for seq in final_outputs)
        padded_outputs = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        
        for i, seq in enumerate(final_outputs):
            padded_outputs[i, :len(seq)] = seq
            
        return padded_outputs
