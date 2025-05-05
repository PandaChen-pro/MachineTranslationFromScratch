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
        Beam search decoding.
        Simplified implementation focusing on core logic. May lack optimizations like
        length normalization or efficient batch handling across beams.
        """
        device = src.device
        batch_size = src.size(0)
        pad_idx = pad_id # Use passed pad_id

        src_padding_mask = (src == pad_idx).to(device)  # [Batch, SrcSeqLen]

        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)

        final_outputs = []

        for b in range(batch_size):
            single_memory = memory[b:b+1].expand(beam_size, -1, -1) # Expand memory for beam
            single_src_padding_mask = src_padding_mask[b:b+1].expand(beam_size, -1) # Expand mask

            k_prev_words = torch.tensor([[bos_id]], dtype=torch.long, device=device) # [1, 1]
            seqs = k_prev_words # Current active hypotheses [beam, seq_len]
            top_k_scores = torch.zeros(1, device=device) # [beam]

            complete_seqs = []
            complete_seqs_scores = []

            step = 1
            while True:
                if step > max_len: # Stop if max length reached
                     break


                tgt_emb = self.positional_encoding(self.tgt_embedding(seqs) * math.sqrt(self.d_model))
                current_beam_size = seqs.size(0)
                current_seq_len = seqs.size(1)


                tgt_padding_mask = (seqs == pad_idx).to(device) # [current_beam_size, current_seq_len]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_seq_len, device=device) # [current_seq_len, current_seq_len]


                if single_memory.size(0) != current_beam_size:
                     single_memory = memory[b:b+1].expand(current_beam_size, -1, -1)
                     single_src_padding_mask = src_padding_mask[b:b+1].expand(current_beam_size, -1)

                # --- Decoder Forward Pass ---
                output = self.transformer_decoder(
                    tgt=tgt_emb,
                    memory=single_memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_padding_mask,
                    memory_key_padding_mask=single_src_padding_mask
                )
                # output: [current_beam_size, current_seq_len, d_model]


                next_token_logits = self.output_layer(output[:, -1, :]) # [current_beam_size, vocab_size]
                next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1) # [current_beam_size, vocab_size]

                cumulative_scores = top_k_scores.unsqueeze(1) + next_token_log_probs

                beam_k = beam_size # Number of candidates to consider for the next step
                if step == 1:
                     flat_scores = cumulative_scores.flatten() # [vocab_size]
                     num_candidates = min(beam_k, flat_scores.numel())
                     top_k_scores, top_k_indices = flat_scores.topk(num_candidates) # [k], [k]
                     beam_indices = torch.zeros_like(top_k_indices) # All came from beam 0
                     next_word_indices = top_k_indices % next_token_log_probs.size(1) # Word indices
                else:
                     flat_scores = cumulative_scores.flatten() # [current_beam_size * vocab_size]
                     num_candidates = min(beam_k, flat_scores.numel())
                     top_k_scores, top_k_indices = flat_scores.topk(num_candidates) # [k], [k]
                     beam_indices = top_k_indices // next_token_log_probs.size(1) # [k] -> Beam index
                     next_word_indices = top_k_indices % next_token_log_probs.size(1) # [k] -> Word index

                next_seqs = []
                next_beam_indices = [] # Track which beams survive
                for i in range(len(top_k_indices)):
                    beam_idx = beam_indices[i]
                    next_word_idx = next_word_indices[i]
                    score = top_k_scores[i]

                    base_seq = seqs[beam_idx]
                    new_seq = torch.cat([base_seq, next_word_idx.view(1)])

                    if next_word_idx == eos_id:
                        complete_seqs.append(new_seq.tolist())
                        complete_seqs_scores.append(score.item())
                    else:
                        next_seqs.append(new_seq)
                        next_beam_indices.append(beam_idx) # Keep track of origin beam (needed if stateful)

                if not next_seqs: # If all sequences ended
                    break

                seqs = torch.stack(next_seqs) # [new_beam_size, new_seq_len]
                survivor_indices = [i for i, next_word_idx in enumerate(next_word_indices) if next_word_idx != eos_id]
                if not survivor_indices: # Check if list is empty
                     if not complete_seqs: # If no survivors and no completed sequences, something is wrong
                          print("Warning: Beam search ended with no completed or surviving sequences.")
                          if seqs.numel() > 0: # Check if seqs was populated before this check
                                best_incomplete_idx = top_k_scores.argmax().item() # Fallback: best score overall
                                final_outputs.append(seqs[best_incomplete_idx]) # Add the best tensor
                          else: # If seqs is empty, maybe add BOS or handle error
                                final_outputs.append(torch.tensor([bos_id, eos_id], device=device)) # Default fallback

                     break # Exit loop if no survivors

                valid_survivor_indices = torch.tensor(survivor_indices, device=device).long()
                if valid_survivor_indices.numel() > 0 :
                    top_k_scores = top_k_scores[valid_survivor_indices]
                else:
                     print("Warning: No valid survivor indices found, breaking beam search.")
                     break


                current_beam_size = seqs.size(0)
                if current_beam_size == 0: # Should be covered by checks above, but safeguard
                     break

                step += 1


            if not complete_seqs: # If no sequence reached EOS
                 if seqs.numel() > 0 : # Check if seqs has elements
                    best_seq_tensor = seqs[top_k_scores.argmax()]
                    final_outputs.append(best_seq_tensor)
                 else:
                    final_outputs.append(torch.tensor([bos_id, eos_id], device=device))

            else:
                best_score_idx = complete_seqs_scores.index(max(complete_seqs_scores))
                best_seq_list = complete_seqs[best_score_idx]
                final_outputs.append(torch.tensor(best_seq_list, device=device))


        max_batch_len = 0
        for seq in final_outputs:
            if seq.numel() > max_batch_len:
                 max_batch_len = seq.numel()

        max_batch_len = max(1, max_batch_len)
        padded_outputs = torch.full((batch_size, max_batch_len), pad_id, dtype=torch.long, device=device)

        for i, seq in enumerate(final_outputs):
            current_len = seq.numel()
            if current_len > 0 : # Ensure sequence is not empty before slicing
                padded_outputs[i, :current_len] = seq

        return padded_outputs