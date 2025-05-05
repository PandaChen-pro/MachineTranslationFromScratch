import argparse
import os
import torch
import wandb
from tqdm import tqdm

from dataset_loader import (
    TranslationDataset,
    build_vocabs,
    get_dataloader,
    Vocabulary
)
from model import TransformerModel
from trainer import Trainer
from util import set_seed, write_translations, get_device, count_parameters

def parse_args():
    parser = argparse.ArgumentParser(description='中英神经机器翻译')
    
    # 数据相关
    parser.add_argument('--train_src', type=str, default='./data/chinese.txt', help='训练集源语言文件')
    parser.add_argument('--train_tgt', type=str, default='./data/english.txt', help='训练集目标语言文件')
    parser.add_argument('--val_src', type=str, default='./data/Niu.dev.txt', help='验证集源语言文件')
    parser.add_argument('--val_tgt', type=str, default='./data/Niu.dev.reference', help='验证集目标语言文件')
    parser.add_argument('--test_src', type=str, default='./data/Niu.test.txt', help='测试集源语言文件')
    parser.add_argument('--test_tgt', type=str, default='Niu.test.reference', help='测试集目标语言文件（可选）')
    
    # 模型相关
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--nhead', type=int, default=8, help='注意力头数')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='编码器层数')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='解码器层数')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout率')
    
    # 训练相关
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--seed', type=int, default=3407, help='随机种子')
    parser.add_argument('--max_len', type=int, default=100, help='最大序列长度')
    
    # 推理相关
    parser.add_argument('--beam_size', type=int, default=5, help='束搜索宽度')
    parser.add_argument('--use_beam_search', action='store_true', help='使用束搜索进行解码')
    
    # 路径相关
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点目录')
    parser.add_argument('--vocab_dir', type=str, default='vocabs', help='词汇表目录')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    
    # wandb相关
    parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb记录训练')
    parser.add_argument('--wandb_project', type=str, default='nmt-transformer', help='wandb项目名')
    parser.add_argument('--wandb_name', type=str, default='zh-en-transformer', help='wandb运行名')
    parser.add_argument('--resume_wandb', action='store_true', help='是否恢复wandb运行')
    
    # 运行模式
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='运行模式')
    parser.add_argument('--resume_training', action='store_true', help='是否从检查点恢复训练')
    
    args = parser.parse_args()
    return args

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 获取设备
    device = get_device()
    print(f"Using device: {device}")
    
    # 创建必要的目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.vocab_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建配置字典
    config = {
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_encoder_layers': args.num_encoder_layers,
        'num_decoder_layers': args.num_decoder_layers,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout,
        'learning_rate': args.lr,
        'clip_grad': args.clip_grad,
        'max_len': args.max_len,
        'beam_size': args.beam_size,
        'use_beam_search': args.use_beam_search,
        'use_wandb': args.use_wandb,
        'wandb_project': args.wandb_project,
        'wandb_name': args.wandb_name,
        'resume_wandb': args.resume_wandb,
        'resume_training': args.resume_training
    }
    
    # 训练模式
    if args.mode == 'train':
        # 加载训练和验证数据
        print("Loading datasets...")
        train_dataset = TranslationDataset(args.train_src, args.train_tgt, args.max_len)
        val_dataset = TranslationDataset(args.val_src, args.val_tgt, args.max_len)
        
        # 构建或加载词汇表
        src_vocab_path = os.path.join(args.vocab_dir, 'src_vocab.pt')
        tgt_vocab_path = os.path.join(args.vocab_dir, 'tgt_vocab.pt')
        
        if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path) and args.resume_training:
            print("Loading vocabularies...")
            src_vocab = Vocabulary.load(src_vocab_path)
            tgt_vocab = Vocabulary.load(tgt_vocab_path)
        else:
            print("Building vocabularies...")
            src_vocab, tgt_vocab = build_vocabs(train_dataset)
            # 保存词汇表
            src_vocab.save(src_vocab_path)
            tgt_vocab.save(tgt_vocab_path)
        
        print(f"Source vocabulary size: {len(src_vocab)}")
        print(f"Target vocabulary size: {len(tgt_vocab)}")
        
        # 创建数据加载器
        train_loader = get_dataloader(train_dataset, args.batch_size, src_vocab, tgt_vocab, shuffle=True)
        val_loader = get_dataloader(val_dataset, args.batch_size, src_vocab, tgt_vocab)
        
        # 创建模型
        if args.resume_training and os.path.exists(os.path.join(args.checkpoint_dir, 'checkpoint.pt')):
            print("Loading model from checkpoint...")
            checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint.pt'))
            model = TransformerModel(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                d_model=args.d_model,
                nhead=args.nhead,
                num_encoder_layers=args.num_encoder_layers,
                num_decoder_layers=args.num_decoder_layers,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout
            )
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Creating new model...")
            model = TransformerModel(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                d_model=args.d_model,
                nhead=args.nhead,
                num_encoder_layers=args.num_encoder_layers,
                num_decoder_layers=args.num_decoder_layers,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout
            )
        
        model = model.to(device)
        print(f"Model has {count_parameters(model):,} trainable parameters")
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            config=config,
            device=device,
            checkpoint_dir=args.checkpoint_dir
        )
        
        # 开始训练
        print("Starting training...")
        trainer.train(args.epochs)
        
    # 测试模式
    elif args.mode == 'test':
        # 加载词汇表
        print("Loading vocabularies...")
        src_vocab_path = os.path.join(args.vocab_dir, 'src_vocab.pt')
        tgt_vocab_path = os.path.join(args.vocab_dir, 'tgt_vocab.pt')
        
        if not os.path.exists(src_vocab_path) or not os.path.exists(tgt_vocab_path):
            raise FileNotFoundError("Vocabulary files not found. Train a model first.")
        
        src_vocab = Vocabulary.load(src_vocab_path)
        tgt_vocab = Vocabulary.load(tgt_vocab_path)
        
        # 加载测试数据
        print("Loading test dataset...")
        test_dataset = TranslationDataset(args.test_src, args.test_tgt, args.max_len)
        test_loader = get_dataloader(test_dataset, args.batch_size, src_vocab, tgt_vocab)
        
        # 加载模型
        print("Loading best model...")
        best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        if not os.path.exists(best_model_path):
            raise FileNotFoundError("Best model not found. Train a model first.")
        
        checkpoint = torch.load(best_model_path)
        model = TransformerModel(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            train_loader=None,
            val_loader=test_loader,
            src_vocab=src_vocab,·
            tgt_vocab=tgt_vocab,
            config=config,
            device=device,
            checkpoint_dir=args.checkpoint_dir
        )
        
        # 评估模型
        print("Evaluating on test set...")
        test_loss, test_bleu = trainer.evaluate()
        print(f"Test Loss: {test_loss:.4f}, Test BLEU: {test_bleu:.4f}")

if __name__ == '__main__':
    main()
