import argparse
import os
import torch
import wandb
from tqdm import tqdm
import nltk
import math # Make sure math is imported if needed elsewhere, though not directly used in this snippet after changes

from dataset_loader import (
    TranslationDataset,
    build_vocabs,
    get_dataloader,
    Vocabulary,
    load_and_split_data
)
from model import TransformerModel
from trainer import Trainer
from util import set_seed, write_translations, get_device, count_parameters # Assumed to exist

# Only call login if using wandb
# Consider calling it inside main() after checking args.use_wandb
# wandb.login() # You might want to handle login errors or make it conditional

def parse_args():
    parser = argparse.ArgumentParser(description='中英神经机器翻译')

    # --- 数据相关 ---
    parser.add_argument('--src_file', type=str, default='./data/chinese.txt', help='源语言文件（中文）')
    parser.add_argument('--tgt_file', type=str, default='./data/english.txt', help='目标语言文件（英文）')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='测试集比例')

    # --- 模型相关 ---
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--nhead', type=int, default=8, help='注意力头数')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='编码器层数')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='解码器层数')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout率')

    # --- 训练相关 ---
    parser.add_argument('--epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=3e-4, help='基础学习率 (Warmup 后的最大学习率)')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='学习率预热步数') 
    parser.add_argument('--clip_grad', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--seed', type=int, default=3407, help='随机种子')
    parser.add_argument('--max_len', type=int, default=100, help='最大序列长度')

    # --- 推理相关 ---
    parser.add_argument('--beam_size', type=int, default=5, help='束搜索宽度')
    parser.add_argument('--use_beam_search', action='store_true', help='使用束搜索进行解码')
    parser.add_argument('--fast_eval', action='store_true', help='使用快速评估模式')
    parser.add_argument('--fast_eval_size', type=int, default=5000, help='快速评估样本数量') 

    # --- 路径相关 ---
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点目录')
    parser.add_argument('--vocab_dir', type=str, default='vocabs', help='词汇表目录')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')

    # --- wandb相关 ---
    parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb记录训练')
    parser.add_argument('--wandb_project', type=str, default='nmt-transformer', help='wandb项目名')
    parser.add_argument('--wandb_name', type=str, default='zh-en-transformer', help='wandb运行名')
    parser.add_argument('--resume_wandb', action='store_true', help='是否恢复wandb运行')

    # --- 运行模式 ---
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

    # 确保nltk的bleu评估所需资源已下载
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading nltk 'punkt' tokenizer...")
        nltk.download('punkt')
        print("Download complete.")

    # --- 创建配置字典 --- (Ensure all necessary args are passed)
    config = {
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_encoder_layers': args.num_encoder_layers,
        'num_decoder_layers': args.num_decoder_layers,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout,
        'learning_rate': args.lr,
        'warmup_steps': args.warmup_steps,          
        'clip_grad': args.clip_grad,
        'max_len': args.max_len,
        'batch_size': args.batch_size,               
        'epochs': args.epochs,                     
        'beam_size': args.beam_size,
        'use_beam_search': args.use_beam_search,
        'use_wandb': args.use_wandb,
        'wandb_project': args.wandb_project,
        'wandb_name': args.wandb_name,
        'resume_wandb': args.resume_wandb,
        'resume_training': args.resume_training,
        'fast_eval': args.fast_eval,
        'fast_eval_size': args.fast_eval_size      
    }

    if args.use_wandb:
        try:
            wandb.login()
        except Exception as e:
            print(f"Wandb login failed: {e}. Disabling wandb.")
            args.use_wandb = False
            config['use_wandb'] = False


    # 加载并分割数据
    print("Loading and splitting data...")
    train_src, train_tgt, val_src, val_tgt, test_src, test_tgt = load_and_split_data(
        args.src_file,
        args.tgt_file,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    # 训练模式
    if args.mode == 'train':
        # 构建数据集
        train_dataset = TranslationDataset(train_src, train_tgt, args.max_len)
        val_dataset = TranslationDataset(val_src, val_tgt, args.max_len)

        # 构建或加载词汇表
        src_vocab_path = os.path.join(args.vocab_dir, 'src_vocab.pt')
        tgt_vocab_path = os.path.join(args.vocab_dir, 'tgt_vocab.pt')

        if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path) and args.resume_training:
            print("Loading vocabularies...")
            src_vocab = Vocabulary.load(src_vocab_path)
            tgt_vocab = Vocabulary.load(tgt_vocab_path)
        else:
            print("Building vocabularies...")
            src_vocab, tgt_vocab = build_vocabs(train_src, train_tgt)
            # 保存词汇表
            src_vocab.save(src_vocab_path)
            tgt_vocab.save(tgt_vocab_path)

        print(f"Source vocabulary size: {len(src_vocab)}")
        print(f"Target vocabulary size: {len(tgt_vocab)}")

        # 创建数据加载器
        train_loader = get_dataloader(train_dataset, args.batch_size, src_vocab, tgt_vocab, shuffle=True)
        val_loader = get_dataloader(val_dataset, args.batch_size, src_vocab, tgt_vocab)

        # 创建模型
        model_path_to_check = os.path.join(args.checkpoint_dir, 'checkpoint.pt')
        if args.resume_training and os.path.exists(model_path_to_check):
             print(f"Resuming training, attempting to load model from {model_path_to_check}...")
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

        # 创建训练器 (Pass the updated config)
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
            raise FileNotFoundError("Vocabulary files not found in {args.vocab_dir}. Train a model first or provide correct path.")

        src_vocab = Vocabulary.load(src_vocab_path)
        tgt_vocab = Vocabulary.load(tgt_vocab_path)
        print(f"Source vocabulary size: {len(src_vocab)}")
        print(f"Target vocabulary size: {len(tgt_vocab)}")

        # 构建测试数据集
        test_dataset = TranslationDataset(test_src, test_tgt, args.max_len)
        test_batch_size = args.batch_size * 2
        test_loader = get_dataloader(test_dataset, test_batch_size, src_vocab, tgt_vocab)

        # 加载模型 (Load the best model)
        print("Loading best model for testing...")
        best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Best model not found at {best_model_path}. Train a model first.")

        # Initialize model structure first
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

        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            state_dict = checkpoint['model_state_dict']
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            print(f"Loaded best model from epoch {checkpoint.get('epoch', 'N/A')} with BLEU {checkpoint.get('best_bleu', 'N/A'):.4f}")
        except Exception as e:
            print(f"Error loading model state dict from {best_model_path}: {e}")
            raise

        print(f"Model has {count_parameters(model):,} parameters")


        # 创建训练器 (Needed for evaluate method, pass None for train_loader)
        trainer = Trainer(
            model=model,
            train_loader=None, 
            val_loader=test_loader, 
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            config=config, 
            device=device,
            checkpoint_dir=args.checkpoint_dir 
        )

        # 评估模型 - 在测试集上使用完整评估
        print("Evaluating on test set...")
        test_loss, test_bleu = trainer.evaluate(fast_eval=False)
        print(f"Test Loss: {test_loss:.4f}, Test BLEU-4: {test_bleu:.4f}")

        # 生成翻译结果并保存
        model.eval() 
        all_translations = []
        all_sources = []
        all_references = []

        print("Generating translations for the test set...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Generating translations"):
                src = batch['src'].to(device)
                src_texts = batch['src_text']
                tgt_texts = batch['tgt_text']

                use_beam = args.use_beam_search
                if use_beam:
                     translated_ids = model.beam_search(src, max_len=args.max_len, beam_size=args.beam_size)
                else:
                     translated_ids = model.greedy_decode(src, max_len=args.max_len)

                translations = [tgt_vocab.decode(ids, skip_special_tokens=True) for ids in translated_ids]

                all_sources.extend(src_texts)
                all_references.extend(tgt_texts)
                all_translations.extend(translations)

        # 保存翻译结果 (Source, Reference, Hypothesis)
        output_file = os.path.join(args.output_dir, 'test_translations.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            for src, ref, hyp in zip(all_sources, all_references, all_translations):
                f.write(f"SRC: {src}\n")
                f.write(f"REF: {ref}\n")
                f.write(f"HYP: {hyp}\n\n")
        print(f"Test translations saved to {output_file}")

if __name__ == '__main__':
    main()