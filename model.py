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

        # 创建自定义TransformerEncoder和TransformerDecoder，以解决mask类型不匹配问题 (Using standard PyTorch layers)
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
            batch_first=True # Ensure batch_first is True
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
        Note: tgt during training is tgt_input (shifted right, excluding last token)
              tgt during greedy/beam decode starts with BOS and grows
        """
        if src_padding_mask is None:
            src_padding_mask = (src == self.pad_idx) 
        if tgt_padding_mask is None:
             tgt_padding_mask = (tgt == self.pad_idx) 

        if tgt_mask is None:
             tgt_len = tgt.size(1)
             tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=tgt.device) 

        if memory_key_padding_mask is None:
            memory_key_padding_mask = src_padding_mask # [Batch, SrcSeqLen]

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


        output = self.output_layer(output) # [Batch, TgtSeqLen, TgtVocabSize]

        return output

    def greedy_decode(self, src, max_len=100, bos_id=2, eos_id=3):
        """优化的贪婪解码，用于推理阶段"""
        batch_size = src.size(0)
        device = src.device
        pad_idx = self.pad_idx 


        src_padding_mask = (src == pad_idx).to(device) #[Batch, SrcSeqLen]

        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)


        ys = torch.ones(batch_size, 1).fill_(bos_id).long().to(device) # [Batch, 1]


        for i in range(max_len - 1): 
            tgt_input = ys 
            tgt_padding_mask = (ys == pad_idx).to(device) 
            tgt_len = ys.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device) 

            tgt_emb = self.positional_encoding(self.tgt_embedding(ys) * math.sqrt(self.d_model))
            out = self.transformer_decoder(
                tgt=tgt_emb,
                memory=memory, # Encoder output
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask, # Likely all False
                memory_key_padding_mask=src_padding_mask # Use the source padding mask here
            )
            # out shape: [Batch, CurrentLen, DimModel]

            last_output = out[:, -1, :] # [Batch, DimModel]
            prob = self.output_layer(last_output) # [Batch, TgtVocabSize]
            next_word = torch.argmax(prob, dim=-1) # [Batch]


            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1) # [Batch, CurrentLen + 1]

            if (next_word == eos_id).all():
                break

        return ys # Return the generated sequences [Batch, FinalLen]


    def beam_search(self, src, max_len=100, beam_size=5, bos_id=2, eos_id=3, pad_id=0):
        """
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