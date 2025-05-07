import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau # No longer used
from torch.optim.lr_scheduler import LambdaLR # Import LambdaLR
import time
import wandb
import os
import json
from tqdm import tqdm
import random
import math # Import math
import sacrebleu

from util import save_checkpoint, load_checkpoint, save_dataset, load_dataset

class LabelSmoothingLoss(nn.Module):
    """
    实现标签平滑的损失函数
    
    Args:
        size: 词汇表大小
        padding_idx: 填充标记的索引，在计算损失时会被忽略
        smoothing: 平滑参数，通常设置为0.1
    """
    def __init__(self, size, padding_idx, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing  # 正确标签的置信度
        self.smoothing = smoothing  # 分配给其他标签的概率
        self.size = size  # 词汇表大小
        self.true_dist = None
        
    def forward(self, x, target):
        """
        计算带标签平滑的损失
        
        Args:
            x: 模型输出的logits，形状为 [batch_size * seq_len, vocab_size]
            target: 目标索引，形状为 [batch_size * seq_len]
            
        Returns:
            平滑后的KL散度损失
        """
        assert x.size(1) == self.size
        true_dist = x.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))  # 将平滑概率均匀分配给所有非正确、非填充的标签
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)  # 将confidence概率分配给正确标签
        true_dist[:, self.padding_idx] = 0  # 填充标记不参与损失计算
        mask = (target == self.padding_idx)
        true_dist.masked_fill_(mask.unsqueeze(1), 0.0)  # 将填充位置的所有概率设为0
        
        # 保存true_dist用于调试
        self.true_dist = true_dist
        
        # 对logits应用log_softmax，然后计算KL散度
        x = torch.log_softmax(x, dim=1)
        non_pad_count = target.numel() - mask.sum().item()
        return self.criterion(x, true_dist) / non_pad_count

class NoamOpt:
    """Noam优化器实现"""
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self.param_groups = optimizer.param_groups  
        
    def step(self):
        "更新参数和学习率"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "实现学习率预热和衰减"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) * \
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    def zero_grad(self):
        self.optimizer.zero_grad()

# --- Learning Rate Scheduler Function ---
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Creates a learning rate scheduler with linear warmup and linear decay.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Linear decay
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class Trainer:
    def __init__(self, model, train_loader, val_loader, src_vocab, tgt_vocab,
                 config, device, checkpoint_dir='checkpoints', data_dir='data'):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The Transformer model.
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation/test set.
            src_vocab (Vocabulary): Source language vocabulary.
            tgt_vocab (Vocabulary): Target language vocabulary.
            config (dict): Configuration dictionary containing hyperparameters.
            device (torch.device): Device to run training on (e.g., 'cuda', 'cpu').
            checkpoint_dir (str): Directory to save checkpoints.
            data_dir (str): Directory to save/load datasets.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        # --- Optimizer ---
        base_optimizer = optim.Adam(
            model.parameters(),
            lr=0,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # 使用Noam优化器
        self.optimizer = NoamOpt(
            model_size=config['d_model'],
            factor=config.get('factor', 1),
            warmup=config.get('warmup_steps', 4000),
            optimizer=base_optimizer
        )

        # --- Loss Function ---
        smoothing = config.get('label_smoothing', 0.1)
        self.criterion = LabelSmoothingLoss(
            size=len(tgt_vocab),
            padding_idx=tgt_vocab.pad_token_id,
            smoothing=smoothing
        )

        # --- Learning Rate Scheduler ---
        self.scheduler = None
        self.num_training_steps = 0
        self.num_warmup_steps = 0

        # Setup scheduler only if in training mode
        if self.train_loader is not None:
            try:
                num_update_steps_per_epoch = len(self.train_loader)
                self.num_training_steps = num_update_steps_per_epoch * config['epochs']
                self.num_warmup_steps = config['warmup_steps']

                if self.num_training_steps <= 0:
                    print(f"Warning: Calculated num_training_steps is {self.num_training_steps}. Scheduler might not work correctly.")

                self.scheduler = get_linear_schedule_with_warmup(
                    base_optimizer,
                    num_warmup_steps=self.num_warmup_steps,
                    num_training_steps=self.num_training_steps
                )
                print(f"Scheduler created: Warmup steps={self.num_warmup_steps}, Total estimated steps={self.num_training_steps}")

            except Exception as e:
                print(f"Error calculating training steps or creating scheduler: {e}")
                self.scheduler = None
        else:
            print("Running in test mode or train_loader not provided. Scheduler not created.")

        # --- Training State ---
        self.start_epoch = 0
        self.best_bleu = 0.0
        self.global_step = 0
        
        # --- 早停机制 ---
        self.patience = config.get('patience', 5)  # 连续5个epoch没有改善则早停
        self.early_stop_counter = 0
        self.should_stop = False

        # --- wandb Configuration ---
        self.use_wandb = config.get('use_wandb', False)
        self.wandb_run = None
        if self.use_wandb:
            self._setup_wandb()

    def _setup_wandb(self):
        """设置wandb配置"""
        wandb_id_file = os.path.join(self.checkpoint_dir, 'wandb_id.json')
        resume_wandb_run = self.config.get('resume_wandb', False) and os.path.exists(wandb_id_file)

        if resume_wandb_run:
            try:
                with open(wandb_id_file, 'r') as f:
                    wandb_data = json.load(f)
                self.wandb_run = wandb.init(
                    project=self.config.get('wandb_project', 'nmt-transformer'),
                    id=wandb_data['id'],
                    resume="must",
                    config=self.config
                )
                print(f"Resuming wandb run with id: {wandb_data['id']}")
            except Exception as e:
                print(f"Failed to resume wandb run: {e}. Initializing new run.")
                resume_wandb_run = False

        if not resume_wandb_run:
            try:
                self.wandb_run = wandb.init(
                    project=self.config.get('wandb_project', 'nmt-transformer'),
                    name=self.config.get('wandb_name', 'zh-en-transformer'),
                    config=self.config
                )
                with open(wandb_id_file, 'w') as f:
                    json.dump({'id': self.wandb_run.id}, f)
                print(f"Initialized new wandb run with id: {self.wandb_run.id}")
            except Exception as e:
                print(f"Failed to initialize wandb: {e}. Disabling wandb.")
                self.use_wandb = False

        if self.use_wandb and self.wandb_run:
            wandb.watch(self.model, log_freq=100)

    def save_training_state(self):
        """保存训练状态"""
        state = {
            'epoch': self.start_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_bleu': self.best_bleu,
            'global_step': self.global_step,
            'config': self.config
        }
        
        # 保存最新检查点
        save_checkpoint(state, False, self.checkpoint_dir)
        
        # 如果是最佳模型，保存一份
        if self.best_bleu > 0:
            save_checkpoint(state, True, self.checkpoint_dir)

    def load_training_state(self):
        """加载训练状态"""
        checkpoint = load_checkpoint(self.checkpoint_dir)
        if checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.best_bleu = checkpoint['best_bleu']
            self.global_step = checkpoint['global_step']
            print(f"Loaded checkpoint from epoch {self.start_epoch}")
            return True
        return False

    def _train_epoch(self, epoch):
        """Trains the model for one epoch."""
        self.model.train() # Set model to training mode
        total_loss = 0
        start_time = time.time()

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}", leave=True)

        for batch_idx, batch in enumerate(progress_bar):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)

            # Prepare decoder input (excluding last token) and target output (excluding first token)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # --- Forward pass ---
            self.optimizer.zero_grad() # Zero gradients before forward pass
            # Model internally creates masks based on padding and causality
            outputs = self.model(src, tgt_input)

            # --- Calculate loss ---
            # Reshape for CrossEntropyLoss: (Batch * SeqLen, VocabSize), (Batch * SeqLen)
            loss = self.criterion(outputs.view(-1, len(self.tgt_vocab)), tgt_output.reshape(-1))

            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at step {self.global_step}. Skipping update.")
                # Consider stopping or adding more debugging here
                continue # Skip backward/optimizer step if loss is NaN

            # --- Backward pass and Optimization ---
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad'])

            self.optimizer.step()

            # --- Update Learning Rate (Step-based Scheduler) ---
            if self.scheduler:
                 self.scheduler.step()

            total_loss += loss.item()
            self.global_step += 1 # Increment global step counter

            # Update progress bar postfix
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.6f}"})

            # Log metrics to wandb periodically
            if self.use_wandb and self.wandb_run and self.global_step % 100 == 0: # Log every 100 steps
                wandb.log({
                    'step_train_loss': loss.item(),
                    'learning_rate': current_lr,
                    # Log global_step directly if needed, wandb usually tracks its own step
                }, step=self.global_step) # Use global_step as custom step counter

        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1} Training finished. Avg Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s")
        return avg_loss

    def evaluate(self, fast_eval=False):
        """评估模型性能"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                # 前向传播计算损失
                outputs = self.model(src, tgt_input)
                loss = self.criterion(outputs.view(-1, len(self.tgt_vocab)), tgt_output.reshape(-1))
                total_loss += loss.item()
                
                # 创建mask
                src_padding_mask = (src == self.src_vocab.pad_token_id).to(self.device)
                tgt_padding_mask = (tgt_input == self.tgt_vocab.pad_token_id).to(self.device)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)

                # 使用beam search进行解码
                if not fast_eval:
                    predictions = self.model.beam_search(
                        src,
                        max_len=self.config.get('max_len', 100),
                        beam_size=self.config.get('beam_size', 5),
                        bos_id=self.tgt_vocab.bos_token_id,
                        eos_id=self.tgt_vocab.eos_token_id,
                        pad_id=self.tgt_vocab.pad_token_id
                    )
                else:
                    predictions = self.model.greedy_decode(
                        src,
                        max_len=self.config.get('max_len', 100),
                        bos_id=self.tgt_vocab.bos_token_id,
                        eos_id=self.tgt_vocab.eos_token_id
                    )

                # 将预测结果转换为文本
                pred_texts = []
                for pred in predictions:
                    tokens = pred.tolist()
                    # 检查tokens是否是嵌套列表，并确保我们处理的是一维列表
                    if isinstance(tokens, list) and any(isinstance(item, list) for item in tokens):
                        # 如果是嵌套列表，仅取第一个子列表
                        tokens = tokens[0] if tokens else []
                    if self.tgt_vocab.eos_token_id in tokens:
                        tokens = tokens[:tokens.index(self.tgt_vocab.eos_token_id)]
                    pred_texts.append(self.tgt_vocab.decode(tokens))

                # 将目标文本转换为文本
                target_texts = []
                for target in tgt_output:
                    tokens = target.tolist()
                    # 检查tokens是否是嵌套列表，并确保我们处理的是一维列表
                    if isinstance(tokens, list) and any(isinstance(item, list) for item in tokens):
                        # 如果是嵌套列表，仅取第一个子列表
                        tokens = tokens[0] if tokens else []
                    if self.tgt_vocab.eos_token_id in tokens:
                        tokens = tokens[:tokens.index(self.tgt_vocab.eos_token_id)]
                    target_texts.append(self.tgt_vocab.decode(tokens))

                all_predictions.extend(pred_texts)
                all_targets.extend(target_texts)

        # 计算平均损失
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        
        # 使用sacrebleu计算BLEU分数
        bleu = sacrebleu.corpus_bleu(all_predictions, [all_targets])
        bleu_score = bleu.score

        if self.use_wandb:
            self.wandb_run.log({
                'val_loss': avg_loss,
                'val_bleu': bleu_score,
                'epoch': self.start_epoch
            })

        return avg_loss, bleu_score

    def train(self, epochs):
        """训练模型"""
        # 尝试加载检查点
        if self.config.get('resume_training', False):
            self._load_checkpoint()

        # 开始训练循环
        print(f"Starting training from epoch {self.start_epoch} to {self.start_epoch + epochs}")
        
        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            if self.should_stop:
                print(f"Early stopping triggered after {epoch} epochs (no improvement for {self.patience} epochs)")
                break
                
            # 训练一个epoch
            train_loss = self._train_epoch(epoch)
            
            # 在验证集上评估
            val_loss, val_bleu = self.evaluate(fast_eval=self.config.get('fast_eval', False))
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val BLEU: {val_bleu:.4f}")
            
            # 记录指标
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_bleu": val_bleu
                }, step=self.global_step)
            
            # 检查是否是最佳模型
            is_best = val_bleu > self.best_bleu
            if is_best:
                self.best_bleu = val_bleu
                self.early_stop_counter = 0  # 重置早停计数器
                print(f"New best model with BLEU: {val_bleu:.4f}")
            else:
                self.early_stop_counter += 1
                print(f"No improvement for {self.early_stop_counter} epochs.")
                if self.early_stop_counter >= self.patience:
                    self.should_stop = True
                    print(f"Early stopping triggered!")
            
            # 保存检查点
            state = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.optimizer.state_dict(),
                'best_bleu': self.best_bleu,
                'global_step': self.global_step
            }
            
            if self.scheduler:
                state['scheduler_state_dict'] = self.scheduler.state_dict()
                
            save_checkpoint(state, is_best, self.checkpoint_dir)
            
        print(f"Training completed. Best BLEU: {self.best_bleu:.4f}")
        
        # 加载最佳模型
        best_checkpoint = load_checkpoint(self.checkpoint_dir, filename='best_model.pt')
        if best_checkpoint:
            self.model.load_state_dict(best_checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {best_checkpoint.get('epoch', 'unknown')} with BLEU: {best_checkpoint.get('best_bleu', 0.0):.4f}")

    def _load_checkpoint(self):
        """Loads checkpoint to resume training."""
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint.pt')
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            # Pass the scheduler to load its state as well
            checkpoint = load_checkpoint(checkpoint_path, self.model, self.optimizer, self.scheduler, self.device)
            if checkpoint:
                # Load states from the returned checkpoint dictionary
                self.start_epoch = checkpoint.get('epoch', 0) # Get epoch, default to 0 if not found
                self.best_bleu = checkpoint.get('best_bleu', 0.0)
                self.global_step = checkpoint.get('global_step', 0)

                # Optimizer and scheduler states are loaded within load_checkpoint util function
                print(f"Resuming training from Epoch {self.start_epoch + 1}, Global Step: {self.global_step}, Best BLEU: {self.best_bleu:.4f}")
            else:
                print("Checkpoint loading failed (load_checkpoint returned None). Starting from scratch.")
                self.start_epoch = 0 # Ensure starting from scratch if loading fails
                self.best_bleu = 0.0
                self.global_step = 0
        else:
            print("No checkpoint found at {checkpoint_path}. Starting training from scratch.")
            self.start_epoch = 0
            self.best_bleu = 0.0
            self.global_step = 0