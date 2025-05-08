import torch
import torch.nn as nn
import math
import torch.nn.functional as F

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

        # 创建自定义TransformerEncoder和TransformerDecoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        decoder_layer = nn.TransformerDecoderLayer( 
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True 
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
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
        self.pad_idx = 0 

    def _reset_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_padding_mask=None, tgt_padding_mask=None,
                memory_key_padding_mask=None):
        """
        前向传播.
        """
        if src_padding_mask is None:
            src_padding_mask = (src == self.pad_idx) 
        if tgt_padding_mask is None:
            tgt_padding_mask = (tgt == self.pad_idx) 

        if tgt_mask is None:
            tgt_len = tgt.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=tgt.device) 

        if memory_key_padding_mask is None:
            memory_key_padding_mask = src_padding_mask 

        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        memory = self.transformer_encoder(
            src=src_emb,
            mask=src_mask, 
            src_key_padding_mask=src_padding_mask 
        )

        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,              
            memory_mask=None,               
            tgt_key_padding_mask=tgt_padding_mask,  
            memory_key_padding_mask=memory_key_padding_mask 
        )

        output = self.output_layer(output) 

        return output

    def greedy_decode(self, src, max_len=100, bos_id=2, eos_id=3):
        """贪婪解码，用于推理阶段"""
        batch_size = src.size(0)
        device = src.device
        pad_idx = self.pad_idx 

        src_padding_mask = (src == pad_idx).to(device) 

        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)

        ys = torch.ones(batch_size, 1).fill_(bos_id).long().to(device) 

        for i in range(max_len - 1): 
            tgt_padding_mask = (ys == pad_idx).to(device) 
            tgt_len = ys.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device) 

            tgt_emb = self.positional_encoding(self.tgt_embedding(ys) * math.sqrt(self.d_model))
            out = self.transformer_decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask
            )

            last_output = out[:, -1, :] 
            prob = self.output_layer(last_output) 
            next_word = torch.argmax(prob, dim=-1) 

            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1) 

            if (next_word == eos_id).all():
                break

        return ys

    def beam_search(self, src, max_len=100, beam_size=5, bos_id=2, eos_id=3, pad_id=0, length_penalty=1.0):
        """
        完整的Beam search解码实现
        
        Args:
            src: 输入序列 [batch_size, src_len]
            max_len: 最大解码长度
            beam_size: 束搜索宽度
            bos_id: 开始标记的ID
            eos_id: 结束标记的ID
            pad_id: 填充标记的ID
            length_penalty: 长度惩罚系数，越大对长序列惩罚越轻
            
        Returns:
            生成的序列 [batch_size, max_len]
        """
        device = src.device
        batch_size = src.size(0)
        pad_idx = pad_id
        
        # 编码源序列
        src_padding_mask = (src == pad_idx).to(device)
        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)
        
        # 为每个样本创建结果容器
        all_beams_results = []
        
        # 对批次中每个样本单独进行Beam Search
        for b in range(batch_size):
            # 提取当前样本的memory和mask
            single_memory = memory[b:b+1].repeat(beam_size, 1, 1)  # [beam_size, src_len, d_model]
            single_src_padding_mask = src_padding_mask[b:b+1].repeat(beam_size, 1)  # [beam_size, src_len]
            
            # 初始化起始序列和分数
            seqs = torch.full((beam_size, 1), bos_id, dtype=torch.long, device=device)  # [beam_size, 1]
            scores = torch.zeros(beam_size, 1, device=device)  # [beam_size, 1]
            
            # 标记是否完成生成
            finish_flag = torch.zeros(beam_size, dtype=torch.bool, device=device)
            
            # 记录完成生成的序列和分数
            final_seqs = []
            final_scores = []
            
            # 开始Beam Search
            for step in range(1, max_len):
                num_effective_beams = (1 - finish_flag.int()).sum().item()
                if num_effective_beams == 0:
                    break
                
                # 解码当前状态
                current_seqs = seqs[~finish_flag]
                current_memory = single_memory[:num_effective_beams]
                current_src_padding_mask = single_src_padding_mask[:num_effective_beams]
                
                # 计算当前mask
                tgt_padding_mask = (current_seqs == pad_idx).to(device)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_seqs.size(1), device=device)
                
                # 计算输出
                tgt_emb = self.positional_encoding(self.tgt_embedding(current_seqs) * math.sqrt(self.d_model))
                out = self.transformer_decoder(
                    tgt=tgt_emb,
                    memory=current_memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_padding_mask,
                    memory_key_padding_mask=current_src_padding_mask
                )
                
                # 获取最后一个时间步的输出并计算概率分布
                last_output = out[:, -1, :]  # [num_effective_beams, d_model]
                logits = self.output_layer(last_output)  # [num_effective_beams, vocab_size]
                log_probs = torch.log_softmax(logits, dim=-1)  # [num_effective_beams, vocab_size]
                
                # 计算候选项的分数
                vocab_size = log_probs.size(-1)
                current_scores = scores[~finish_flag]  # [num_effective_beams, 1]
                
                # 添加新token的分数，并进行长度惩罚
                # 长度惩罚公式: (5 + length)^length_penalty / (5 + 1)^length_penalty
                length_penality_factor = ((5 + step) ** length_penalty) / ((5 + 1) ** length_penalty)
                candidate_scores = (current_scores + log_probs) / length_penality_factor
                
                # 平铺分数以便选择top-k
                candidate_scores = candidate_scores.view(-1)  # [num_effective_beams * vocab_size]
                
                # 选择k个最佳候选项，k值为未完成序列数量与beam_size的最小值
                k = min(beam_size, candidate_scores.size(0))
                top_values, top_indices = candidate_scores.topk(k)
                
                # 计算序列索引和词汇表索引
                beam_indices = top_indices // vocab_size  # 哪个束
                token_indices = top_indices % vocab_size  # 哪个token
                
                # 创建新一轮的序列
                new_seqs = []
                new_scores = []
                new_finish_flags = []
                
                # 更新序列
                for i in range(k):
                    beam_idx = beam_indices[i]
                    token_idx = token_indices[i]
                    beam_score = top_values[i]
                    
                    # 获取新序列
                    new_seq = torch.cat([current_seqs[beam_idx], token_idx.unsqueeze(0)], dim=0)
                    
                    # 检查是否结束
                    if token_idx == eos_id:
                        final_seqs.append(new_seq)
                        # 计算实际长度（跳过最后的EOS标记）
                        real_length = (new_seq != pad_id).sum().item() - 1
                        # 对完成的序列应用长度惩罚
                        final_score = beam_score * ((5 + real_length) ** length_penalty) / ((5 + 1) ** length_penalty)
                        final_scores.append(final_score.item())
                    else:
                        new_seqs.append(new_seq)
                        new_scores.append(beam_score)
                        new_finish_flags.append(False)
                
                # 如果没有足够的新序列，则补充已完成的序列
                while len(new_seqs) < beam_size and final_seqs:
                    worst_finished_idx = final_scores.index(min(final_scores))
                    new_seqs.append(final_seqs.pop(worst_finished_idx))
                    new_scores.append(final_scores.pop(worst_finished_idx))
                    new_finish_flags.append(True)
                
                # 如果仍然没有足够的序列，填充复制前一轮的序列
                while len(new_seqs) < beam_size:
                    # 复制最好的序列
                    best_idx = new_scores.index(max(new_scores))
                    new_seqs.append(new_seqs[best_idx])
                    new_scores.append(new_scores[best_idx])
                    new_finish_flags.append(new_finish_flags[best_idx])
                
                # 更新序列和分数
                max_seq_len = max(seq.size(0) for seq in new_seqs)
                updated_seqs = torch.full((beam_size, max_seq_len), pad_id, dtype=torch.long, device=device)
                for i, seq in enumerate(new_seqs):
                    updated_seqs[i, :seq.size(0)] = seq
                
                seqs = updated_seqs
                scores = torch.tensor(new_scores, device=device).unsqueeze(1)
                finish_flag = torch.tensor(new_finish_flags, dtype=torch.bool, device=device)
            
            # 添加剩余未完成的序列到最终结果
            for i in range(beam_size):
                if not finish_flag[i]:
                    final_seqs.append(seqs[i])
                    final_scores.append(scores[i].item())
            
            # 选择得分最高的序列
            if final_seqs:
                best_idx = final_scores.index(max(final_scores))
                best_seq = final_seqs[best_idx]
            else:
                # 如果没有完成的序列，返回第一个序列
                best_seq = seqs[0]
            
            all_beams_results.append(best_seq)
        
        # 将所有结果统一长度
        max_length = max(seq.size(0) for seq in all_beams_results)
        final_results = torch.full((batch_size, max_length), pad_id, dtype=torch.long, device=device)
        
        for i, seq in enumerate(all_beams_results):
            final_results[i, :seq.size(0)] = seq
            
        return final_results
=======
        优化的beam search解码实现
        
        Args:
            src: 源语言输入 [batch_size, src_len]
            max_len: 最大生成长度
            beam_size: beam大小
            bos_id: 开始标记的ID
            eos_id: 结束标记的ID
            pad_id: 填充标记的ID
            
        Returns:
            生成的序列列表
        """
        device = src.device
        batch_size = src.size(0)
        
        # 我们将分别处理每个batch item
        results = []
        
        for batch_idx in range(batch_size):
            # 提取单个batch item
            src_item = src[batch_idx:batch_idx+1]  # 保持维度 [1, src_len]
            
            # 准备encoder输出
            src_padding_mask = (src_item == pad_id).to(device)
            src_emb = self.positional_encoding(self.src_embedding(src_item) * math.sqrt(self.d_model))
            memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)
            
            # 初始化beams
            start_token = torch.tensor([[bos_id]], dtype=torch.long, device=device)
            beams = [(start_token, 0.0)]
            final_beams = []
            
            for _ in range(max_len - 1):
                candidates = []
                
                # 对每个beam进行扩展
                for beam_seq, beam_score in beams:
                    if beam_seq[0, -1].item() == eos_id:
                        final_beams.append((beam_seq, beam_score))
                        continue
                        
                    # 准备decoder输入
                    tgt_padding_mask = (beam_seq == pad_id).to(device)
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(beam_seq.size(1)).to(device)
                    
                    # 获取decoder输出
                    tgt_emb = self.positional_encoding(self.tgt_embedding(beam_seq) * math.sqrt(self.d_model))
                    out = self.transformer_decoder(
                        tgt=tgt_emb,
                        memory=memory,
                        tgt_mask=tgt_mask,
                        tgt_key_padding_mask=tgt_padding_mask,
                        memory_key_padding_mask=src_padding_mask
                    )
                    
                    # 获取最后一个时间步的预测
                    last_output = out[:, -1, :]
                    logits = self.output_layer(last_output)
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    # 获取top-k个候选
                    topk_log_probs, topk_indices = log_probs.topk(beam_size, dim=-1)
                    
                    # 扩展beam
                    for k in range(beam_size):
                        candidate_seq = torch.cat([beam_seq, topk_indices[:, k:k+1]], dim=1)
                        candidate_score = beam_score - topk_log_probs[:, k].item()  # 使用item()而不是mean()
                        candidates.append((candidate_seq, candidate_score))
                
                # 选择最好的beam_size个候选
                candidates.sort(key=lambda x: x[1])
                beams = candidates[:beam_size]
                
                # 检查是否所有beam都生成了EOS
                if all(beam_seq[0, -1].item() == eos_id for beam_seq, _ in beams):
                    break
            
            # 将未完成的beam添加到final_beams
            final_beams.extend(beams)
            
            # 选择得分最高的beam
            final_beams.sort(key=lambda x: x[1])
            best_beam = final_beams[0][0]  # 只取最好的一个beam
            results.append(best_beam)  # 添加到结果列表
        
        return results
