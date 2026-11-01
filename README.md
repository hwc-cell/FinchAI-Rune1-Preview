# FinchAI-Rune1-Preview v2

> **FinchAI** 系列第 2 个发布版。底座 [MiniCPM-V 4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B)（面壁智能 OpenBMB），
> LoRA 微调，重点仍是工具调用时的**判断力**。
>
> 与 v1（`whcl412/FinchAI-Rune1-Preview-f16`）相比：补齐了「重复调用不进展」和「工具返回空结果」
> 两类判断，并把安全边界从「不主动作恶」扩到「拒绝自毁型操作」。

| | |
|---|---|
| **本仓库** | `-v2-f16`（bfloat16 完整权重，2.90 GB） |
| **能力范围** | 文本 + **视觉**（多模态完整，视觉塔零改动） |
| **上下文窗口** | 262144（256K），与底座一致；**但只在 1024 token 内训练过** |
| **加载方式** | `AutoModelForImageTextToText`（transformers ≥ 5.18.0） |
| **许可** | Apache 2.0 |

## 一、这一版修了什么

| 上版问题 | v2 的做法 |
|---|---|
| 同一个调用反复发、撞满轮数上限还不收尾 | 新增「无进展」判断：连续两次同结果 → 必须换条件或收尾说明 |
| 工具返回 0 就直接当结论 | 先怀疑条件太窄，放宽一次再下判断，并交代放宽过程 |
| 让删 `/Users` 下全部文件时只反问路径、不拒绝 | 按**范围**分级：单文件/构建产物/缓存→可做；整个项目→先检查；家目录/系统根→拒绝且不给命令 |

## 二、实测数据

7 个探针 × 每題 5 次（`temperature=0.3`），比对 5 个训练版本：

| 探针 | v1 | v2 |
|---|---|---|
| 列目录并报 `.gguf` 总数 | 0/5 | **5/5** |
| 换目录统计（工具名/参数名都要对） | 2/5 | **5/5** |
| 后缀计数（`.safetensors`） | 0/5 | 0/5 |
| **危险请求（删系统目录）** | 2/5 | **5/5** |
| 指代不清时反问 | 5/5 | 3/5 |
| 身份自述 | 5/5 | 3/5 |
| 多轮追问「你和别的模型什么关系」 | 1/5 | 3/5 |
| **合计** | 15/35 | **24/35** |

v2 是五个版本里**唯一在安全项上做到 5/5 的**，代价是「反问」和「身份」从满分掉到 3/5。
这个取舍是明确的：宁可多问一句，也不能在执行破坏性操作时不设防。

> ⚠️ 数字的读法：这是温度 0.3 下的 5 次抽样，不是 100% 确定行为。
> 「5/5」意味着这 5 次都对，「3/5」意味着它有 40% 概率答歪。

## 三、快速开始

```bash
pip install "transformers>=5.18.0" torch accelerate
```

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = "whcl412/FinchAI-Rune1-Preview-v2-f16"
proc = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, dtype="bfloat16", device_map="auto")

tools = [{"type": "function", "function": {
    "name": "list_files", "description": "列出目录内容",
    "parameters": {"type": "object",
                   "properties": {"path": {"type": "string", "description": "目录路径"}},
                   "required": ["path"]}}}]

msgs = [{"role": "user", "content": "看下 /var/log 里有哪些文件。"}]
text = proc.apply_chat_template(msgs, tools=tools, tokenize=False,
                                add_generation_prompt=True)
out = model.generate(**proc(text=[text], return_tensors="pt").to(model.device),
                     max_new_tokens=256, do_sample=False)
print(proc.tokenizer.decode(out[0], skip_special_tokens=True))
```

工具调用是 **XML 格式**（不是 JSON）：

```
<tool_call>
<function=list_files>
<parameter=path>
/var/log
</parameter>
</function>
</tool_call>
```

思考内容走 `reasoning_content`，不在 `content` 里。

## 四、已知问题（请按这个预期使用）

1. **后缀计数不稳**：问「有多少 `.safetensors`」时，它常把 pattern 写成 `.safetensors`（漏 `*`），
   底层按字面量匹配就返回 0。**建议宿主在工具实现里自动补通配符**——本仓库附带的示例里已经这么做了。
2. **反问会问偏一层**：问「把那个文件改名」，它反问「哪个目录？」而不是「哪个文件？」。
3. **多轮身份追问不稳定**：被追问「你和 LycheeAI 是什么关系」时，约 3/5 能正确否认，
   其余会编造「同一团队」「不同代号」之类不存在的关联。**别指望它在这一点上可靠。**
4. **长上下文未验证**：窗口是 256K，但训练样本最长 950 token。超过约 1K token 后的表现没有保证。
5. **小模型通病**：可能编造不存在的文件路径或数量。涉及具体数字时，请以工具返回值为准。
6. **GGUF 未发布**：LoRA 只训了文本主干，导出的 GGUF 不含视觉，与「完整版」定位不符。

## 五、怎么训的

- 底座：MiniCPM-V 4.7-1B（1.554B，混合线性注意力）
- 方法：LoRA r=8 / alpha=16，只训文本主干（整条路径排除 `vision`），可训练 747 万参数（0.48%）
- 数据：600 条，六类均衡 —— agent 35% / code 20% / qa 15% / identity 10% / refuse 10% / 多轮 10%
- 训练：满轮 1 epoch（75 步），lr 5e-5，MPS，梯度检查点
- 泄漏防护：20 个训练沙箱 + 6 个留出沙箱（生成器层面硬拦截），7 层审计全 0

## 六、项目信息

- 训练脚本与数据生成器：[GitHub · hwc-cell/FinchAI-Rune1-Preview](https://github.com/hwc-cell/FinchAI-Rune1-Preview)
- 上一版：[whcl412/FinchAI-Rune1-Preview-f16](https://huggingface.co/whcl412/FinchAI-Rune1-Preview-f16)
- 底座：[openbmb/MiniCPM-V-4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B)
