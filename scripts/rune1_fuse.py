#!/usr/bin/env python3
"""FinchAI-Rune1-Preview · 融合脚本（LoRA 合并进基座，出 bf16 safetensors）

## 为什么是 torch 而不是 mlx_lm
MiniCPM-V 4.7-1B 的架构 `minicpmv4_7` 不被 mlx_lm 支持（连最新 main 都没有），
所以 FinchAI 整条链都是 torch 路线。**产物是标准 safetensors + bf16**，
transformers 和 llama.cpp 都能直接加载。

## 命名铁律
目录/仓库名必须同时暴露「格式」和「精度」—— LycheeAI 时期踩过：
把 4bit 放在无后缀的主仓库、把无损放 `-f16`，结果「MLX + 无损」这个组合在仓库列表里搜不到。

## 在 CPU 上合并
合并是纯权重运算，不需要 GPU；走 MPS 反而会和别的进程抢统一内存。

用法：
  python rune1_fuse.py
"""
import glob
import os
import time

for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(k, None)
os.environ['HF_HUB_OFFLINE'] = '1'

MDL = '/Users/hwc/LycheeAI/MiniCPM-V-4.7-1B'
ADAPTER = '/Users/hwc/LycheeAI/adapters-finchai-rune1'
OUT = '/Users/hwc/LycheeAI/FinchAI-Rune1-Preview-f16'

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

t0 = time.time()
print('加载基座（bf16，CPU，不走 MPS）...', flush=True)
base = AutoModelForImageTextToText.from_pretrained(
    MDL, dtype=torch.bfloat16, low_cpu_mem_usage=True)
print(f'  {time.time()-t0:.1f}s', flush=True)

print('加载 adapter 并合并 ...', flush=True)
model = PeftModel.from_pretrained(base, ADAPTER)
model = model.merge_and_unload()          # LoRA 权重并进基座，得到一个普通模型
print(f'  已合并（{time.time()-t0:.1f}s）', flush=True)

print(f'保存到 {OUT} ...', flush=True)
model.save_pretrained(OUT, safe_serialization=True)
AutoProcessor.from_pretrained(MDL).save_pretrained(OUT)

files = sorted(os.path.basename(f) for f in glob.glob(OUT + '/*'))
tot = sum(os.path.getsize(os.path.join(OUT, f)) for f in files if os.path.isfile(f))
print(f'  文件 {len(files)} 个：{files}', flush=True)
print(f'  合计 {tot/2**30:.2f} GiB', flush=True)
print(f'\n✓ 融合完成，用时 {(time.time()-t0)/60:.1f} 分钟', flush=True)
print(f'  产物 → {OUT}', flush=True)
