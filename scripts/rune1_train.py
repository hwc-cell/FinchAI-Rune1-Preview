#!/usr/bin/env python3
"""FinchAI-Rune1-Preview · 训练脚本（transformers + PEFT + MPS）

## 为什么不是 mlx_lm
MiniCPM-V 4.7-1B 的架构是 MiniCPMV4_7ForConditionalGeneration，
mlx_lm 连最新 main 都不支持（只有 minicpm / minicpm3），所以走 torch 路线。

## 三个已实测的关键设置（别改）
1. **cap 0.70**：`set_per_process_memory_fraction`。实测 985 token ✓ / 1033 ✗，本进程峰值 ≈6.5GB
2. **梯度检查点必开**：不开的话 608 token 就 OOM
3. **LoRA 按整条路径排除视觉塔**：用后缀匹配会误伤 vision 的同名层

## loss 只算 assistant 部分
把非 assistant 的 token 全部 mask 成 -100。否则模型会去拟合用户提问和工具返回，
那是纯粹的浪费，还会让它倾向于"自己续写用户的话"。

用法：
  python rune1_train.py --smoke        # 跑 3 步验证通路
  python rune1_train.py                # 正式训练
"""
import argparse
import gc
import json
import os
import sys
import time

for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(k, None)
os.environ['HF_HUB_OFFLINE'] = '1'
# ★ 不要动 PYTORCH_MPS_HIGH_WATERMARK_RATIO：
#   =0.0 会取消显存上限（拖死系统）；=1.0 会低于默认低水位 1.4 直接报错。
#   真正管用的是下面 torch.mps.set_per_process_memory_fraction()。

DATA = '/Users/hwc/LycheeAI/finchai_rune1_train.jsonl'
MDL = '/Users/hwc/LycheeAI/MiniCPM-V-4.7-1B'
OUT = '/Users/hwc/LycheeAI/adapters-finchai-rune1'
STATE = '/Users/hwc/LycheeAI/adapters-finchai-rune1/train_state.json'
SAVE_EVERY = 20
DEFAULT_EPOCHS = 2.0
MEM_CAP = 1.00      # ★ 必须 1.00（= Apple 的 recommended max working set，≈11.84 GiB）
                    #   实测 0.90 时：跑到 step 55 遇到 950 token 的样本，
                    #   backward 瞬间 peak = allocated 8.54 + other 1.88 + 需要 0.91 = 11.33 GiB
                    #   > 上限 10.66 GiB → OOM。**数据里 31.5% 的样本是 900~950 token，
                    #   跳不掉，只能把上限给够。** 1.00 时上限 11.84 GiB，仍远低于玮晨给的 15GB。
MAX_LEN = 1024

ap = argparse.ArgumentParser()
ap.add_argument('--smoke', action='store_true', help='只跑几步验证通路')
ap.add_argument('--epochs', type=float, default=DEFAULT_EPOCHS)
ap.add_argument('--accum', type=int, default=8)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--cap', type=float, default=MEM_CAP)
ap.add_argument('--resume', action='store_true',
                help='从 OUT 里已有的 adapter + train_state.json 续训')
args = ap.parse_args()

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model, PeftModel

print(f"torch {torch.__version__}  MPS={torch.backends.mps.is_available()}", flush=True)
torch.mps.set_per_process_memory_fraction(args.cap)
print(f"MPS 单进程上限 {args.cap*100:.0f}%", flush=True)


def free_pct():
    out = os.popen('memory_pressure 2>/dev/null | grep -i "free percentage"').read()
    return int(''.join(c for c in out if c.isdigit()) or 0)


print(f"启动前系统空闲 {free_pct()}%", flush=True)


# ───────────────────────── 数据 ─────────────────────────
class ChatDS(Dataset):
    """只对 assistant 段算 loss。

    做法：把渲染后的整段文本按 '<|im_start|>assistant\\n' 切开，
    属于 assistant 的片段给出真实 label，其余片段 label = -100。
    分段单独 tokenize 再拼接 —— 比事后找 token 下标稳，不会踩边界错位。
    """

    A_START = '<|im_start|>assistant\n'
    A_END = '<|im_end|>\n'

    def __init__(self, path, tok, max_len=MAX_LEN):
        self.rows = [json.loads(l) for l in open(path, encoding='utf-8')]
        self.tok = tok
        self.max_len = max_len
        self.skipped = 0

    def __len__(self):
        return len(self.rows)

    def _segments(self, text):
        """返回 [(片段文本, 是否算 loss), ...]"""
        segs, pos = [], 0
        while True:
            i = text.find(self.A_START, pos)
            if i < 0:
                if pos < len(text):
                    segs.append((text[pos:], False))
                break
            if i > pos:
                segs.append((text[pos:i], False))
            j = text.find(self.A_END, i)
            j = len(text) if j < 0 else j + len(self.A_END)
            segs.append((text[i:j], True))
            pos = j
        return segs

    def __getitem__(self, idx):
        s = self.rows[idx]
        text = self.tok.apply_chat_template(s['messages'], tools=s.get('tools'),
                                            tokenize=False, add_generation_prompt=False)
        ids, labels = [], []
        for seg, train in self._segments(text):
            t = self.tok(seg, add_special_tokens=False)['input_ids']
            ids += t
            labels += (t if train else [-100] * len(t))
        if len(ids) > self.max_len:
            ids, labels = ids[:self.max_len], labels[:self.max_len]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels)}


def collate(batch):
    L = max(len(b["input_ids"]) for b in batch)
    pad = 0
    I = torch.full((len(batch), L), pad, dtype=torch.long)
    Lb = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        I[i, :n] = b["input_ids"]
        Lb[i, :n] = b["labels"]
    return {"input_ids": I, "labels": Lb,
            "attention_mask": (I != 0).long()}


# ───────────────────────── 模型 ─────────────────────────
print("加载 tokenizer / processor ...", flush=True)
proc = AutoProcessor.from_pretrained(MDL)
tok = proc.tokenizer

print("加载模型（bf16，不走 float32 中转）...", flush=True)
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(MDL, dtype=torch.bfloat16,
                                                    low_cpu_mem_usage=True)
print(f"  {time.time()-t0:.1f}s", flush=True)

# ★ LoRA 只打文本主干（按整条路径排除视觉塔）
SUF = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj',
       'in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b', 'out_proj'}
targets = sorted({n.split('.')[-1] for n, m in model.named_modules()
                  if isinstance(m, torch.nn.Linear) and 'vision' not in n
                  and n.split('.')[-1] in SUF})
if args.resume and os.path.exists(os.path.join(OUT, 'adapter_model.safetensors')):
    print(f"↻ 从已有 adapter 续训：{OUT}", flush=True)
    model = PeftModel.from_pretrained(model, OUT, is_trainable=True)
else:
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                             target_modules=targets, task_type="CAUSAL_LM"))
model.print_trainable_parameters()
model.gradient_checkpointing_enable()
# ★ 不要调 enable_input_require_grads()：实测它让梯度检查点保留更多激活，
#   第一批反传就因为要不到 896MB 的 lm_head 梯度而 OOM。

model = model.to('mps', dtype=torch.bfloat16)
torch.mps.synchronize(); gc.collect()
print(f"搬到 MPS，占用 {torch.mps.current_allocated_memory()/2**30:.2f} GiB\n", flush=True)
model.train()

ds = ChatDS(DATA, tok)
print(f"数据集 {len(ds)} 条", flush=True)

# 先看一条的 loss 覆盖情况（确认 mask 生效）
sample = ds[0]
n_lbl = sum(1 for x in sample['labels'].tolist() if x != -100)
print(f"首条样本：{len(sample['input_ids'])} token，其中算 loss 的 {n_lbl} "
      f"（{n_lbl/len(sample['input_ids'])*100:.0f}%）", flush=True)
if n_lbl == 0:
    sys.exit("✗ 没有任何 token 参与 loss，mask 逻辑有问题")

from torch.optim import AdamW
opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

steps_per_epoch = max(1, len(ds) // args.accum)
total = int(steps_per_epoch * args.epochs)
print(f"计划：{steps_per_epoch} 步/epoch × {args.epochs} epoch = {total} 步"
      f"（accum={args.accum}）\n", flush=True)

os.makedirs(OUT, exist_ok=True)

# ── 断点续训：顺序可复现（每 epoch 固定 seed），按已消费样本数精确跳过 ──
seen, step, micro = 0, 0, 0
low_hits = 0
skipped = 0
if args.resume and os.path.exists(STATE):
    st = json.load(open(STATE))
    seen = st.get('seen', 0)
    step = st.get('step', 0)
    micro = st.get('micro', 0)
    print(f"↻ 上次进度 step {step}/{st.get('total', total)}（已消费 {seen} 条样本），继续\n",
          flush=True)


def save_all(seen, step, micro):
    model.save_pretrained(OUT)
    json.dump({'seen': seen, 'step': step, 'micro': micro, 'total': total},
              open(STATE, 'w'))


run_loss, t_start = 0.0, time.time()
done = False
for epoch in range(int(args.epochs) + 1):
    g = torch.Generator().manual_seed(1000 + epoch)   # 固定种子 → 顺序可复现
    loader = DataLoader(ds, batch_size=1, shuffle=True, generator=g, collate_fn=collate)
    for batch in loader:
        if done:
            break
        seen += 1
        if seen <= step * args.accum:      # 已训过的样本，跳过（保证 step 与 seen 对齐）
            micro = max(micro, 1)
            continue
        try:
            out = model(input_ids=batch['input_ids'].to('mps'),
                        attention_mask=batch['attention_mask'].to('mps'),
                        labels=batch['labels'].to('mps'))
            loss_val = out.loss.item()
            (out.loss / args.accum).backward()
            del out
        except RuntimeError as e:
            if 'out of memory' not in str(e):
                raise
            # ★ 单条超长样本爆内存，不该让整个训练陪葬：
            #   清干净显存、记一笔、跳过它，继续跑后面的。
            torch.mps.empty_cache()
            gc.collect()
            skipped += 1
            if skipped <= 3 or skipped % 10 == 0:
                print(f"    ⚠ 跳过一条样本（第 {skipped} 条，"
                      f"{batch['input_ids'].shape[1]} token）", flush=True)
            continue
        # ★ 每个 micro 都清一次，不只在 step 边界：
        #   样本长度在 75~950 token 之间跳（相差 12 倍），只在 step 边界清，
        #   分配器会留着一堆大小不匹配的空洞，长样本要 900MB 连续块时就要不到。
        torch.mps.empty_cache()
        run_loss += loss_val
        micro += 1

        if micro % args.accum == 0:
            # ★ 系统级安全闸：连续两次低于 10% 才停。
            #   实测 memory_pressure 的空闲率在 14%~48% 之间上下跳，
            #   单次低点只是采样噪声，用单次判定会把好好的训练误杀。
            fp = free_pct()
            if fp < 10:
                low_hits += 1
                if low_hits >= 2:
                    save_all(seen, step, 0)
                    print(f"\n⚠ 系统空闲连续偏低（{fp}%），主动停止并保存"
                          f"（已完成 {step} 步，可 --resume 续训）", flush=True)
                    sys.exit(3)
            else:
                low_hits = 0
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            # ★ 每步清一次缓存：样本长度差异大，不清的话 MPS 池子会被切碎
            torch.mps.empty_cache()
            if step % 5 == 0 or step <= 3:
                el = time.time() - t_start
                print(f"  step {step:>4}/{total}  loss {run_loss/micro:.4f}  "
                      f"{el/(step*args.accum):.2f}s/样本  MPS "
                      f"{torch.mps.current_allocated_memory()/2**30:.2f}G  空闲 {free_pct()}%",
                      flush=True)
                run_loss = 0.0; micro = 0
            if step % SAVE_EVERY == 0:
                save_all(seen, step, 0)
                print(f"    · 已保存（step {step}/{total}）", flush=True)
            # ★★ 到 total 就停，必须显式判！外层是 range(int(epochs) + 1)，
            #   epochs=2.0 时它会给出 3 个 epoch —— 少了这一句就会多训一整轮：
            #   2026-10-17 实测跑了 253 步（计划 142），loss 被压到 0.0038，
            #   过拟合到把工具参数名都教坏了（详见 rune1_eval.py 的判据注释）。
            if step >= total:
                done = True
                break
            if args.smoke and step >= 3:
                done = True
                break

model.save_pretrained(OUT)
tok.save_pretrained(OUT)
json.dump({'seen': seen, 'step': step, 'micro': micro, 'total': total, 'finished': True},
          open(STATE, 'w'))
print(f"\n✓ 训练结束，{step} 步，用时 {(time.time()-t_start)/60:.1f} 分钟", flush=True)
if skipped:
    print(f"  （因显存不足跳过 {skipped} 条超长样本）", flush=True)
print(f"  adapter → {OUT}", flush=True)
