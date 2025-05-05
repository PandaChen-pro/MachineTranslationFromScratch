# 从零开始实现Transformer机器翻译
## 解压数据
```shell
tar -xzf /data/coding/MachineTranslationFromScratch/data/sample.tar.gz 
```
## 训练模型
```shell
python start.py --mode train --train_src chinese.txt --train_tgt english.txt --val_src Niu.dev.txt --val_tgt Niu.dev.reference --use_wandb --wandb_project nmt-zh-en
```
## 恢复中断训练
```shell
python start.py --mode train --resume_training --resume_wandb
```
## 测试集上评估
```shell
python start.py --mode test --test_src Niu.test.txt --test_tgt Niu.test.reference
```
