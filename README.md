# 从零开始实现Transformer机器翻译
## 解压数据
```shell
tar -xzf /data/coding/MachineTranslationFromScratch/data/sample.tar.gz 
```
## 训练模型
```shell
python start.py --mode train --train_src chinese.txt --train_tgt english.txt --val_src Niu.dev.txt --val_tgt Niu.dev.reference --use_wandb --wandb_project nmt-zh-en
```
```shell
python start.py --mode train --src_file ./data/chinese.txt --tgt_file ./data/english.txt --use_wandb --use_beam_search
```
快速评估：
```shell
python start.py --mode train --fast_eval --use_wandb
python start.py --mode train --fast_eval --fast_eval_size 100
```
## 恢复中断训练
```shell
python start.py --mode train --resume_training --resume_wandb
```
快速评估：
```shell
python start.py --mode train --resume_training --resume_wandb --fast_eval
```
## 测试集上评估
```shell
python start.py --mode test --test_src Niu.test.txt --test_tgt Niu.test.reference
```
