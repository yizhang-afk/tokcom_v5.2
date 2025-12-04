# TokCom v5.2

基于 VAE 和扩散模型的文本压缩与生成系统。

## 概述

TokCom v5.2 是一个两阶段训练的模型：
1. **VAE (变分自编码器)**: 将 token 序列压缩为潜在向量
2. **扩散模型**: 使用 Flow Matching 学习潜在空间的去噪

### 模型架构

| 参数 | 值 |
|------|-----|
| Tokenizer | `meta-llama/CodeLlama-7b-hf` |
| 序列长度 | 1024 tokens |
| Chunk 大小 | 4 tokens/latent vector |
| Chunk 数量 | 256 latent vectors/序列 |
| 隐藏层维度 | 896 |
| Transformer 层数 | 24 (VAE 和扩散模型相同) |

## 安装依赖

```bash
pip install torch transformers datasets tqdm wandb hf_transfer
```

## 项目结构

```
TokCom_v5.2_zy/
├── model.py        # 模型定义 (VAE, DenoisedModel)
├── dataset.py      # WikiText 数据集处理
├── train.py        # 训练脚本 (含 wandb 日志)
├── inference.py    # 推理和生成脚本
├── config.py       # 模型配置 (JSON)
└── README.md
```

## 训练

### 快速开始

```bash
python train.py
```

### 训练配置

在 `train.py` 中修改 `TrainConfig`:

```python
class TrainConfig:
    # 数据集
    dataset_name: str = "wikitext-2-raw-v1"
    tokenizer_path: str = "meta-llama/CodeLlama-7b-hf"
    max_length: int = 1024
    chunk_size: int = 4

    # 训练参数
    batch_size: int = 8
    num_workers: int = 4

    # 阶段一: VAE 训练
    vae_epochs: int = 5
    vae_lr: float = 1e-4

    # 阶段二: 扩散模型训练
    diffusion_epochs: int = 10
    diffusion_lr: float = 3e-5
    noise_std: float = 0.05

    # 采样配置
    sample_every: int = 1      # 每 N 个 epoch 采样一次
    num_sample_steps: int = 50 # 去噪采样步数
    num_samples: int = 2       # 采样数量

    # Wandb
    wandb_project: str = "TokCom-v5.2"
```

### 训练阶段

**阶段一: VAE 训练**
- 训练编码器-解码器重建 token 序列
- 损失函数: 交叉熵重建损失
- 输出: `checkpoints/best_vae.pt`

**阶段二: 扩散模型训练**
- 冻结 VAE，训练去噪模型
- 使用 Flow Matching: `x_t = (1-t)*x0 + t*x1`
- 损失函数: 通过冻结的 VAE decoder 计算交叉熵
- 输出: `checkpoints/best_diffusion.pt`

### Wandb 日志

训练脚本会记录以下内容到 wandb：

**Batch 级别**
- `vae/batch_loss`: VAE 每个 batch 的损失
- `diffusion/batch_ce_loss`: 扩散模型每个 batch 的交叉熵损失
- `diffusion/batch_mse_loss`: 扩散模型每个 batch 的 MSE 损失

**Epoch 级别**
- 训练/验证损失
- 学习率
- 最佳验证损失

**采样**
- VAE 重建样本表格
- 扩散模型生成样本表格
- 去噪重建对比表格

## 推理

### 模式一: 从噪声生成

从随机噪声生成新文本:

```bash
python inference.py \
    --checkpoint checkpoints/best_diffusion.pt \
    --mode generate \
    --num_steps 50 \
    --batch_size 2
```

### 模式二: VAE 重建

通过 VAE 编码-解码重建文本:

```bash
python inference.py \
    --checkpoint checkpoints/best_diffusion.pt \
    --mode reconstruct \
    --input_text "你要重建的文本内容"
```

### 模式三: 去噪重建

通过扩散模型去噪重建 (encode -> 加噪 -> 去噪 -> decode):

```bash
python inference.py \
    --checkpoint checkpoints/best_diffusion.pt \
    --mode denoise_reconstruct \
    --input_text "你要重建的文本内容" \
    --noise_level 0.5 \
    --num_steps 50
```

`noise_level` 参数说明:
- `0.0`: 纯噪声，完全丢失原始信息
- `0.5`: 50% 噪声混合
- `0.9`: 10% 噪声，保留大部分原始信息
- `1.0`: 无噪声，等同于 VAE 重建

### 模式四: 编码为潜在向量

将文本编码为潜在向量并保存:

```bash
python inference.py \
    --checkpoint checkpoints/best_diffusion.pt \
    --mode encode \
    --input_text "你要编码的文本内容" \
    --output_file latents.pt
```

### 模式五: 条件生成

给定 prompt，生成后续内容 (prompt encode -> 与噪声拼接 -> 去噪 -> decode):

```bash
python inference.py \
    --checkpoint checkpoints/best_diffusion.pt \
    --mode conditional \
    --prompt "The family of four was " \
    --num_generate_chunks 64 \
    --num_steps 50
```

条件生成流程:
1. 将 prompt 编码为 latent vectors
2. 生成噪声 latent vectors 作为待生成部分
3. 将 prompt latents 和噪声 latents 拼接
4. 通过 diffusion model 去噪（保持 prompt 部分不变，只对噪声部分去噪）
5. 解码得到完整序列（prompt + 生成内容）

### 推理参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--checkpoint` | checkpoint 文件路径 | 必需 |
| `--mode` | 推理模式 | `generate` |
| `--input_text` | 输入文本 (reconstruct/denoise_reconstruct/encode) | None |
| `--prompt` | 提示文本 (conditional 模式) | None |
| `--num_chunks` | 生成的 chunk 数量 (generate 模式) | 256 |
| `--num_generate_chunks` | prompt 后生成的 chunk 数量 (conditional 模式) | 64 |
| `--num_steps` | 去噪步数 | 50 |
| `--noise_level` | 噪声水平 (denoise_reconstruct 模式) | 0.5 |
| `--batch_size` | 批次大小 | 1 |
| `--device` | 设备 (cuda/cpu) | 自动检测 |
| `--output_file` | 输出文件路径 | None |

## Checkpoint 格式

checkpoint 包含以下内容:

```python
{
    'epoch': int,              # 保存时的 epoch
    'stage': str,              # 'vae' 或 'diffusion'
    'val_loss': float,         # 验证集损失
    'model_args': {            # 模型架构参数
        'hidden_size': 896,
        'num_hidden_layers': 24,
        'num_attention_heads': 14,
        'num_key_value_heads': 2,
        'intermediate_size': 4864,
        'vocab_size': int,
        'omni_token_id': int,
        ...
    },
    'config': {                # 训练/推理配置
        'tokenizer_path': str,
        'max_length': 1024,
        'chunk_size': 4,
        'noise_std': 0.05,
        ...
    },
    'vae_model_state_dict': dict,       # VAE 模型权重
    'denoised_model_state_dict': dict,  # 扩散模型权重
}
```

## 代码中加载模型

```python
from model import TokComVAE, DenoisedModel, ModelArgs
from inference import (
    load_checkpoint,
    generate,
    reconstruct,
    denoise_reconstruct,
    encode,
    decode
)

# 加载模型
vae_model, denoised_model, model_args, config = load_checkpoint(
    "checkpoints/best_diffusion.pt",
    device="cuda"
)

# 从噪声生成
token_ids = generate(
    vae_model, denoised_model,
    num_chunks=256,
    chunk_size=4,
    hidden_size=model_args.hidden_size,
    noise_std=config['noise_std'],
    num_steps=50,
    batch_size=1,
    device="cuda"
)

# 解码为文本
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
print(text)
```

## 模型细节

### VAE 编码器
- 输入: Token IDs `(batch, chunk_size)`
- 输出: 潜在向量 `(batch, hidden_size)`
- 使用最后一个位置作为输出向量

### VAE 解码器
- 输入: 潜在向量 `(batch, hidden_size)`
- 输出: Logits `(batch, chunk_size, vocab_size)`
- 使用 Omni token 进行序列生成

### 扩散模型
- 架构: 24 层 Transformer + adaLN-Zero 条件化
- 条件: 时间步嵌入
- 预测目标: 从噪声输入 `x_t` 预测干净数据 `x1`

### Flow Matching
- 前向过程: `x_t = (1-t)*x0 + t*x1`
- `x0`: 噪声 ~ N(0, noise_std²)
- `x1`: VAE 编码器输出的干净潜在向量
- `t`: 时间步 ~ U[0, 1]

## 许可证

MIT License
