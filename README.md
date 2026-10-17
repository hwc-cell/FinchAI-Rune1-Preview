# FinchAI-Rune1-Preview

> 首个 **FinchAI** 系列模型。底座 [MiniCPM-V 4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B)（面壁智能 OpenBMB），
> 用 LoRA 微调而成，**重点是工具调用时的「判断力」**。

| | |
|---|---|
| **本仓库** | `-f16`（bfloat16 完整权重，2.9 GB） |
| **能力范围** | 文本 + **视觉**（多模态完整） |
| **加载方式** | `AutoModelForImageTextToText`（transformers ≥ 5.18.0） |
| **许可** | Apache 2.0 |

**这是当前唯一发布的版本，也是能力最完整的一份**：含视觉塔，可继续微调、可二次量化。
（GGUF 转换已验证可行，但导出的 GGUF 不含视觉，与「最完整」的定位不符，故暂不单独发布。）

---

## 一、这是什么

一个 1.55B 的本地小助手，专攻**什么时候该调工具、该调哪个**。

大多数小模型的工具调用只学会了「看到工具就调」或者「照抄工具名」。
Rune1-Preview 训练的重点是**判断**：

| 场景 | 期望行为 |
|---|---|
| 用户给了完整参数 | **直接调**，不要再问 |
| 用户漏了必填参数 | **反问**，不要瞎猜一个去试 |
| 两个工具都「能看文件」 | 按语义**选对**那个（按内容筛 vs 列目录） |
| 问题根本不需要工具 | **不调**，直接回答 |
| 调用报错了 | 读错误、**改对再来**，而不是放弃或硬编 |

## 二、验收数据（微调前 vs 微调后）

同一份基座权重，靠切换 adapter 跑两遍，27 条探针：

| 组 | 微调后 | 基座 |
|---|---|---|
| **判断力**（该不该调 / 要不要反问 / 选哪个） | **9/9** | 3/9 |
| 身份 | **3/3** | 0/3 |
| 留出沙箱（训练时从未见过的工具） | **6/6** | 6/6 |
| 已训沙箱 | **6/6** | 6/6 |
| 通用（问答 / 代码 / 拒绝） | **3/3** | 3/3 |
| **合计** | **27/27** | 21/27 |

**净增量 = 判断力 +6、身份 +3，合计 +9 项。**

其中有代表性的对比：

```
问：帮我看下目录里有什么文件。          （没给路径）
  微调后 → 「哪个目录？给我具体路径，我就去看。」   ✓ 反问
  基座   → 直接拿一个猜的路径去调了                  ✗

问：D:/inbox 里哪些文件提到 error？     （两个工具都能碰文件）
  微调后 → 调用搜索工具 seek_*                       ✓ 选对
  基座   → 调用列表工具 catalog_*                    ✗ 选错（但看起来"能跑"）
```

### 关于「留出沙箱」

**留出沙箱 = 训练时结构上不可能产出样本的沙箱**（生成器里直接抛异常拦截）。
它存在的意义是给「数据有没有泄漏」一个硬指标：
模型在那些**从未见过的工具名、路径、参数 schema** 上也能正确调用，
说明它学的是「读工具定义、按语义选」，而不是把训练时见过的名字背下来。

## 三、快速开始

### transformers

```bash
pip install "transformers>=5.18.0" torch accelerate
```

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = "whcl412/FinchAI-Rune1-Preview-f16"
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

**工具调用的输出格式是 XML：**

```
<tool_call>
<function=list_files>
<parameter=path>
/var/log
</parameter>
</function>
</tool_call>
```

思考内容走 `reasoning_content`（不在 `content` 里）。

### Ollama / llama.cpp

**尚未发布。** GGUF 转换本身已验证可行（llama.cpp 自 2026-10-17 起支持
`MiniCPMV4_7ForConditionalGeneration`），但导出的 GGUF **不含视觉能力**
（LoRA 只训了文本主干），与「最完整版本」的定位不符，因此暂不单独发布。

⚠️ **两个必须知道的限制**：

1. 我们**没有发布 MLX 4bit 版本** —— 底座架构 `minicpmv4_7` 不被 `mlx_lm` 支持
   （连最新 main 都没有）。Apple Silicon 用户请走 transformers，MPS 实测可用
   （训练峰值内存 6.5 GB）。
2. 全链路（加载 / 微调 / 融合）**只在 Apple Silicon 的 MPS 上验证过**，未在 CUDA 上实测。

## 四、能力边界

**擅长**
- 本地 agent 的「工具选择大脑」——判断该不该调、调哪个、参数够不够
- 读一份没见过的工具定义并按它调用
- 简短、不谄媚的中文/英文回答

**不擅长**
- 复杂多步推理和算法设计（1B 容量限制）
- 长文档理解（训练时序列 ≤950 token）
- 视觉任务——底座支持图像输入，但**本版 LoRA 只训了文本主干**，视觉能力未经强化

**实测性能**（MacBook Pro M4 / 16 GB，MPS）

| 项目 | 数值 |
|---|---|
| 权重体积 | 2.9 GB（bfloat16） |
| 加载后常驻内存 | ~3.1 GB |
| 训练/推理峰值 | ~6.5 GB |
| 训练速度 | 2.2~3.5 s/样本（≤950 token） |

**已知问题**
- 输出里偶发游离的 `</think>` 标签，宿主接入时建议过滤
- 思考文案带模板生成痕迹（如中英混排），扩规模前会换更自然的生成方式

## 五、怎么训的（简述）

**沙箱式数据**：每个训练样本来自一个程序化生成的「沙箱」，包含独立的
工具名、参数 schema、路径风格、错误消息风格。任务是**真的去读工具定义**才能做对。

**Agent 分五个子类型**，这是模型价值的来源：

| 子类型 | 教什么 | 占比 |
|---|---|---|
| `single` | 标准两步调用 | 25% |
| `recover` | 撞错 → 读错误 → 改对重试 | 20% |
| `disambig` | 两个相似工具里按语义选对 | 20% |
| `no_tool` | 有工具但这个问题不该用 | 20% |
| `ask_back` / `ask_direct` | 缺参数才反问，给了参数就直接调 | 15% |

**防泄漏五层**：命名空间物理隔离 / 生成器硬拦截留出沙箱 / 渲染后 token 预算 /
渲染后文本审计 / 非 Agent 样本禁含沙箱标识符 —— 外加第六层「参数名必须属于该工具定义」。

**防遗忘**（增量训练时）：数据保持完整六类配比 + 学习率减半 + 只训 0.5 epoch。

## 六、局限与许可

- 底座 MiniCPM-V 4.7-1B 为 Apache 2.0，本模型沿用
- 这是 **Preview** 版本：判断力已达标（9/9），但句式泛化和拒绝场景仍在扩充
- 训练数据是**合成的**，用于教学/研究/自建 agent 均可，请自测后再用于生产

## 七、项目信息

```
权重    HuggingFace  whcl412/FinchAI-Rune1-Preview-f16
        ModelScope   whcl412/FinchAI-Rune1-Preview-f16
代码    GitHub       hwc-cell/FinchAI-Rune1-Preview
```
GitHub 仓库含训练脚本、数据生成器、验收脚本与 LoRA adapter。
训练数据为**程序化合成**（600 条），不使用任何真实用户数据。

---

*FinchAI 系列 · 训练于 Apple Silicon（transformers + PEFT + MPS）*
