# 从零开始实现Transformer机器翻译
## 解压数据
```shell
tar -xzf /data/coding/MachineTranslationFromScratch/data/sample.tar.gz 
```
## 训练模型
```shell
python start.py --mode train --use_wandb
```
推荐使用快速评估🛠️：
```shell
python start.py --mode train  --use_wandb --fast_eval
```
## 恢复中断训练
```shell
python start.py --mode train --resume_training --resume_wandb 
```
## 测试集上评估
```shell
python start.py --mode test --use_beam_search
```
模型下载链接：https://drive.google.com/file/d/1NFsf_vvMs3V_D7nfhUZf9TBxPDbM8Lg6/view?usp=sharing

Greedy Search BLEU4:14.4061

Beam Search BLEU4（beam size=3）: Test Loss: 2.1525, Test BLEU-4: 22.1030 (36minutes)
