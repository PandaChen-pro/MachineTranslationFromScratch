import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import wandb
import os
import json
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm
import random

from util import save_checkpoint, load_checkpoint

class Trainer:
    def __init__(self, model, train_loader, val_loader, src_vocab, tgt_vocab, 
                 config, device, checkpoint_dir='checkpoints'):
        """
        初始化训练器
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # 确保检查点目录存在
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 初始化优化器
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'], 
            betas=(0.9, 0.98), 
            eps=1e-9
        )
        
        # 初始化损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_token_id)
        
        # 初始化学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.7, 
            patience=config.get('lr_patience', 2),
            verbose=True
        )
        
        # 训练状态
        self.start_epoch = 0
        self.best_bleu = 0.0
        self.global_step = 0
        
        # wandb配置
        self.use_wandb = config.get('use_wandb', True)
        if self.use_wandb:
            # 检查是否需要恢复wandb运行
            if config.get('resume_wandb', False) and os.path.exists(os.path.join(checkpoint_dir, 'wandb_id.json')):
                with open(os.path.join(checkpoint_dir, 'wandb_id.json'), 'r') as f:
                    wandb_data = json.load(f)
                    wandb.init(
                        project=config.get('wandb_project', 'nmt-transformer'),
                        name=config.get('wandb_name', 'zh-en-transformer'),
                        id=wandb_data['id'],
                        resume="must"
                    )
            else:
                # 新建wandb运行
                wandb.init(
                    project=config.get('wandb_project', 'nmt-transformer'),
                    name=config.get('wandb_name', 'zh-en-transformer'),
                    config=config
                )
                # 保存wandb运行ID以供后续恢复
                with open(os.path.join(checkpoint_dir, 'wandb_id.json'), 'w') as f:
                    json.dump({'id': wandb.run.id}, f)
                    
            # 记录模型图
            wandb.watch(model)
            
    def train(self, epochs):
        """训练模型"""
        # 如果有检查点，加载之
        if self.config.get('resume_training', False):
            self._load_checkpoint()
            
        # 从上次中断的位置继续训练
        for epoch in range(self.start_epoch, epochs):
            # 训练一个epoch
            train_loss = self._train_epoch(epoch)
            
            # 验证
            val_loss, bleu = self.evaluate(fast_eval=True)
            
            # 更新学习率
            self.scheduler.step(bleu)
            
            # 记录指标
            if self.use_wandb:
                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'bleu': bleu,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, BLEU: {bleu:.4f}")
            
            # 保存检查点
            is_best = bleu > self.best_bleu
            if is_best:
                self.best_bleu = bleu
            
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_bleu': self.best_bleu,
                'global_step': self.global_step,
            }, is_best, self.checkpoint_dir)
    
    def _train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(progress_bar):
            # 移至GPU
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            
            # 准备输入和目标
            tgt_input = tgt[:, :-1]  # 去掉最后一个token
            tgt_output = tgt[:, 1:]  # 去掉第一个token
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input)
            
            # 计算损失
            loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('clip_grad', 1.0))
            
            # 更新参数
            self.optimizer.step()
            
            # 更新统计
            total_loss += loss.item()
            self.global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix(loss=loss.item())
            
            # 记录到wandb
            if self.use_wandb and batch_idx % self.config.get('log_interval', 50) == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'global_step': self.global_step
                })
                
        return total_loss / len(self.train_loader)
    
    def evaluate(self, fast_eval=False):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_references = []
        all_hypotheses = []
        
        # 如果启用快速评估，则只评估指定数量的批次
        eval_size = self.config.get('fast_eval_size', 100)  # 默认使用100个样本进行快速评估
        
        with torch.no_grad():
            # 如果快速评估且数据集较大，则随机选择一部分
            if fast_eval and len(self.val_loader) > eval_size / self.config.get('batch_size', 64):
                # 随机采样一些批次
                batch_indices = list(range(len(self.val_loader)))
                random.shuffle(batch_indices)
                sampled_indices = batch_indices[:eval_size]
                eval_loader = [self.val_loader.dataset[i] for i in sampled_indices]
                eval_loader = torch.utils.data.DataLoader(
                    eval_loader, 
                    batch_size=self.config.get('batch_size', 64),
                    collate_fn=self.val_loader.collate_fn
                )
            else:
                eval_loader = self.val_loader
            
            for batch in tqdm(eval_loader, desc="Evaluating"):
                # 移至GPU
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                
                # 准备参考翻译
                references = batch['tgt_text']
                
                # 准备输入和目标
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # 前向传播
                output = self.model(src, tgt_input)
                
                # 计算损失
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                total_loss += loss.item()
                
                # 生成翻译 - 训练中使用贪婪搜索以加速评估
                if fast_eval:
                    translations = self._generate_translations(src, use_beam_search=False)
                else:
                    translations = self._generate_translations(src, use_beam_search=self.config.get('use_beam_search', True))
                
                # 收集参考和假设翻译
                for i, (ref, hyp) in enumerate(zip(references, translations)):
                    all_references.append([ref.split()])
                    all_hypotheses.append(hyp.split())
        
        # 计算BLEU分数
        smoothie = SmoothingFunction().method1
        bleu = corpus_bleu(
            all_references, 
            all_hypotheses, 
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie
        )
        
        return total_loss / len(eval_loader), bleu
    
    def _generate_translations(self, src, use_beam_search=None):
        """生成翻译"""
        # 如果未指定，则使用配置中的设置
        if use_beam_search is None:
            use_beam_search = self.config.get('use_beam_search', True)
            
        if use_beam_search:
            # 使用束搜索
            beam_size = self.config.get('beam_size', 5)
            beam_outputs = self.model.beam_search(
                src, 
                beam_size=beam_size,
                max_len=self.config.get('max_len', 100),
                bos_id=self.tgt_vocab.bos_token_id,
                eos_id=self.tgt_vocab.eos_token_id
            )
            translations = [self.tgt_vocab.decode(output) for output in beam_outputs]
        else:
            # 使用贪婪搜索
            greedy_outputs = self.model.greedy_decode(
                src,
                max_len=self.config.get('max_len', 100),
                bos_id=self.tgt_vocab.bos_token_id,
                eos_id=self.tgt_vocab.eos_token_id
            )
            translations = [self.tgt_vocab.decode(output) for output in greedy_outputs]
            
        return translations
    
    def _load_checkpoint(self):
        """加载检查点"""
        checkpoint = load_checkpoint(self.checkpoint_dir)
        if checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.best_bleu = checkpoint['best_bleu']
            self.global_step = checkpoint['global_step']
            print(f"Resuming from epoch {self.start_epoch}, best BLEU: {self.best_bleu:.4f}")
        else:
            print("No checkpoint found, starting from scratch.")
