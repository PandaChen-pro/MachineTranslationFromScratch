import torch
import torch.nn as nn
import torch.optim as optim
import time
import wandb
import os
import json
import math
import sacrebleu
from tqdm import tqdm
import random

# 导入工具函数
from util import save_checkpoint, load_checkpoint

class LabelSmoothingLoss(nn.Module):
    """
    实现标签平滑的损失函数
    """
    def __init__(self, size, padding_idx, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = (target == self.padding_idx)
        true_dist.masked_fill_(mask.unsqueeze(1), 0.0)
        
        self.true_dist = true_dist
        
        x = torch.log_softmax(x, dim=1)
        non_pad_count = target.numel() - mask.sum().item()
        return self.criterion(x, true_dist) / non_pad_count

class NoamOpt:
    """Noam优化器实现，修复版"""
    def __init__(self, model_size, factor=2.0, warmup=4000):
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self.step_num = 0
        self.optimizer = None
        
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def step(self):
        """更新参数并调整学习率"""
        self.step_num += 1
        lr = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self.optimizer.step()
        
    def rate(self, step=None):
        """计算Noam学习率，确保有正值"""
        if step is None:
            step = self.step_num
        # 避免step=0的除零错误
        step = max(1, step)
        # 按照Transformer论文的公式计算学习率
        return self.factor * (self.model_size ** (-0.5) * 
                             min(step ** (-0.5), step * (self.warmup ** (-1.5))))
        
    def zero_grad(self):
        """清除梯度"""
        self.optimizer.zero_grad()
        
    def get_lr(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']

class Trainer:
    def __init__(self, model, train_loader, val_loader, src_vocab, tgt_vocab,
                 config, device, checkpoint_dir='checkpoints'):
        """
        初始化Trainer
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        # --- 创建Noam优化器 ---
        base_lr = config.get('learning_rate', 3e-4)
        warmup_steps = config.get('warmup_steps', 4000)
        factor = base_lr * (model.d_model ** 0.5) * (warmup_steps ** 0.5)

        # 创建基础Adam优化器
        self.base_optimizer = optim.Adam(
            model.parameters(),
            lr=base_lr,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=1e-4
        )
        
        # 创建Noam优化器包装器
        self.optimizer = NoamOpt(
            model_size=model.d_model,
            factor=factor,  # 使用计算出的factor
            warmup=warmup_steps
        )
        self.optimizer.set_optimizer(self.base_optimizer)

        # --- 损失函数 ---
        smoothing = config.get('label_smoothing', 0.1)
        self.criterion = LabelSmoothingLoss(
            size=len(tgt_vocab),
            padding_idx=tgt_vocab.pad_token_id,
            smoothing=smoothing
        )

        # --- 训练状态 ---
        self.start_epoch = 0
        self.best_bleu = 0.0
        self.global_step = 0

        # --- wandb配置 ---
        self.use_wandb = config.get('use_wandb', False)
        self.wandb_run = None
        if self.use_wandb:
            wandb_id_file = os.path.join(checkpoint_dir, 'wandb_id.json')
            resume_wandb_run = config.get('resume_wandb', False) and os.path.exists(wandb_id_file)

            if resume_wandb_run:
                try:
                    with open(wandb_id_file, 'r') as f:
                        wandb_data = json.load(f)
                    self.wandb_run = wandb.init(
                        project=config.get('wandb_project', 'nmt-transformer'),
                        id=wandb_data['id'],
                        resume="must",
                        config=config
                    )
                    print(f"Resuming wandb run with id: {wandb_data['id']}")
                except Exception as e:
                    print(f"Failed to resume wandb run: {e}. Initializing new run.")
                    resume_wandb_run = False

            if not resume_wandb_run:
                try:
                    self.wandb_run = wandb.init(
                        project=config.get('wandb_project', 'nmt-transformer'),
                        name=config.get('wandb_name', 'zh-en-transformer'),
                        config=config
                    )
                    # 保存wandb运行ID以便将来恢复
                    with open(wandb_id_file, 'w') as f:
                        json.dump({'id': self.wandb_run.id}, f)
                    print(f"Initialized new wandb run with id: {self.wandb_run.id}")
                except Exception as e:
                    print(f"Failed to initialize wandb: {e}. Disabling wandb.")
                    self.use_wandb = False

            # 监视模型（如果wandb运行处于活动状态）
            if self.use_wandb and self.wandb_run:
                wandb.watch(model, log_freq=100)

    def _train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        start_time = time.time()

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}", leave=True)

        for batch_idx, batch in enumerate(progress_bar):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)

            # 准备解码器输入和目标输出
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(src, tgt_input)

            # 计算损失
            loss = self.criterion(outputs.view(-1, len(self.tgt_vocab)), tgt_output.reshape(-1))

            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at step {self.global_step}. Skipping update.")
                continue

            # 反向传播和优化
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad'])

            self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

            # 更新进度条后缀
            current_lr = self.optimizer.get_lr()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.6f}"})

            # 定期记录指标到wandb
            if self.use_wandb and self.wandb_run and self.global_step % 100 == 0:
                wandb.log({
                    'step_train_loss': loss.item(),
                    'learning_rate': current_lr,
                }, step=self.global_step)

        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1} Training finished. Avg Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s")
        return avg_loss

    def evaluate(self, fast_eval=False):
        """使用sacrebleu评估模型"""
        self.model.eval()
        total_loss = 0
        hypotheses = []
        references = []

        # 确定是否使用快速评估
        use_fast_eval = self.config.get('fast_eval', False) and fast_eval
        fast_eval_size = self.config.get('fast_eval_size', 1000)

        num_batches_to_eval = len(self.val_loader)
        eval_desc = "Evaluating (Full)"
        if use_fast_eval:
            if self.val_loader.batch_size and self.val_loader.batch_size > 0:
                num_batches_to_eval = max(1, math.ceil(fast_eval_size / self.val_loader.batch_size))
                eval_desc = f"Evaluating (Fast ~{fast_eval_size} samples / {num_batches_to_eval} batches)"
            else:
                print("Warning: Cannot determine batch size for fast evaluation. Evaluating all batches.")
                num_batches_to_eval = len(self.val_loader)

        # 添加特定句子的翻译测试
        test_sentence = "中国科学院成都计算机应用研究所"
        test_translation = self._translate_test_sentence(test_sentence)

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader, desc=eval_desc, leave=False)):
                if i >= num_batches_to_eval:
                    break

                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)

                # 计算损失
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                outputs = self.model(src, tgt_input)
                loss = self.criterion(outputs.view(-1, len(self.tgt_vocab)), tgt_output.reshape(-1))

                if not torch.isnan(loss):
                    total_loss += loss.item()
                else:
                    print(f"Warning: NaN loss detected during evaluation batch {i}. Skipping batch loss.")

                # 生成翻译
                use_beam = self.config.get('use_beam_search', False)
                max_len = self.config.get('max_len', 100)
                beam_size = self.config.get('beam_size', 5)

                if use_beam:
                    translated_ids = self.model.beam_search(
                        src, 
                        max_len=max_len, 
                        beam_size=beam_size,
                        length_penalty=self.config.get('length_penalty', 1.0)
                    )
                else:
                    translated_ids = self.model.greedy_decode(src, max_len=max_len)

                # 解码并准备BLEU计算
                for j in range(translated_ids.size(0)):
                    # 解码假设（预测）
                    hyp_text = self.tgt_vocab.decode(translated_ids[j], skip_special_tokens=True)
                    # 解码参考
                    ref_text = self.tgt_vocab.decode(tgt[j], skip_special_tokens=True)

                    hypotheses.append(hyp_text)
                    references.append(ref_text)

        # 使用sacrebleu计算BLEU分数
        bleu = 0.0
        if hypotheses and references:
            try:
                # 使用sacrebleu计算BLEU-4分数
                bleu_score = sacrebleu.corpus_bleu(
                    hypotheses, 
                    [references], 
                    tokenize='13a',  # 使用标准的BLEU分词器13a
                    lowercase=True,  # 转换为小写以保持一致性
                    force=True  # 忽略分词警告
                )
                bleu = bleu_score.score  # 使用原始的0-100范围的BLEU分数
            except Exception as e:
                print(f"Error in BLEU calculation: {e}")
                bleu = 0.0
        else:
            print("Warning: No hypotheses or references generated for BLEU calculation.")

        avg_loss = total_loss / num_batches_to_eval if num_batches_to_eval > 0 else 0

        # 记录特定句子的翻译结果到wandb
        if self.use_wandb and self.wandb_run:
            translation_table = wandb.Table(columns=["Source", "Translation"])
            translation_table.add_data(test_sentence, test_translation)
            wandb.log({
                'test_sentence_translation': translation_table,
            }, commit=False)  # 与其他指标一起记录

        # 保存检查点（仅在训练模式下）
        if self.train_loader is not None:
            current_epoch = getattr(self, 'epoch_count', self.start_epoch)
            is_best = bleu > self.best_bleu
            if is_best:
                self.best_bleu = bleu
                save_checkpoint({
                    'epoch': current_epoch,
                    'global_step': self.global_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.base_optimizer.state_dict(),
                    'best_bleu': self.best_bleu,
                }, True, self.checkpoint_dir, filename='best_model.pt')
                print(f"New best model saved with BLEU: {self.best_bleu:.4f}")

            # 始终保存最新的检查点（用于恢复）
            save_checkpoint({
                'epoch': current_epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.base_optimizer.state_dict(),
                'best_bleu': self.best_bleu,
            }, False, self.checkpoint_dir, filename='checkpoint.pt')

        return avg_loss, bleu

    def _translate_test_sentence(self, sentence):
        """翻译指定的测试句子并返回结果"""
        oov_chars = []
        for char in sentence:
            if char not in self.src_vocab.token2idx:
                oov_chars.append(char)
        
        if oov_chars:
            print(f"警告：句子中有{len(oov_chars)}个字符不在词汇表中: {' '.join(oov_chars)}")
        # 将句子编码为模型输入格式
        tokens = self.src_vocab.encode(sentence)
        print(f"分词结果: {tokens}")
        src_tensor = torch.tensor([tokens], device=self.device)
        
        # 确定解码方法
        use_beam = self.config.get('use_beam_search', False)
        max_len = self.config.get('max_len', 100)
        beam_size = self.config.get('beam_size', 5)
        
        # 翻译句子
        with torch.no_grad():
            if use_beam:
                translated_ids = self.model.beam_search(
                    src_tensor, 
                    max_len=max_len, 
                    beam_size=beam_size,
                    length_penalty=self.config.get('length_penalty', 1.0)
                )
            else:
                translated_ids = self.model.greedy_decode(src_tensor, max_len=max_len)
        
        # 解码翻译结果
        translation = self.tgt_vocab.decode(translated_ids[0], skip_special_tokens=True)
        
        print(f"测试句子翻译: '{sentence}' -> '{translation}'")
        return translation

    def train(self, epochs):
        """主训练循环"""
        # 如果恢复训练，则加载检查点
        if self.config.get('resume_training', False):
            self._load_checkpoint()

        print(f"Starting training from Epoch {self.start_epoch + 1}...")

        for epoch in range(self.start_epoch, epochs):
            self.epoch_count = epoch + 1

            # 训练一个epoch
            train_loss = self._train_epoch(epoch)

            # 在验证集上评估
            val_loss, bleu = self.evaluate(fast_eval=self.config.get('fast_eval', False))

            # 记录指标
            log_data = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_bleu': bleu,
                'learning_rate': self.optimizer.get_lr(),
                'best_val_bleu': self.best_bleu,
            }
            if self.use_wandb and self.wandb_run:
                wandb.log(log_data)

            print(f"Epoch {epoch+1}/{epochs} Completed. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val BLEU: {bleu:.4f}, Best Val BLEU: {self.best_bleu:.4f}")

        # 训练循环结束后结束wandb运行
        if self.use_wandb and self.wandb_run:
            print("Finishing wandb run...")
            wandb.finish()

    def _load_checkpoint(self):
        """加载检查点以恢复训练"""
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint.pt')
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 加载模型状态
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # 处理多GPU训练保存的模型
                if list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
                self.model.load_state_dict(state_dict)
            
            # 加载优化器状态
            if 'optimizer_state_dict' in checkpoint and self.base_optimizer:
                try:
                    self.base_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    print(f"Warning: Failed to load optimizer state: {e}")
            
            # 加载训练状态
            self.start_epoch = checkpoint.get('epoch', 0) + 1  # 下一个epoch
            self.best_bleu = checkpoint.get('best_bleu', 0.0)
            self.global_step = checkpoint.get('global_step', 0)
            
            print(f"Resuming training from Epoch {self.start_epoch}, Global Step: {self.global_step}, Best BLEU: {self.best_bleu:.4f}")
        else:
            print(f"No checkpoint found at {checkpoint_path}. Starting training from scratch.")
            self.start_epoch = 0
            self.best_bleu = 0.0
            self.global_step = 0