#!/usr/bin/env python3
"""FinchAI-Rune1 · 均衡版训练数据生成器

玮晨的要求（2026-10-16 23:29）：
  · 不要只训练 Agent 能力，均衡搭配
  · 折中，自己看着办
  · 峰值内存不要到 15GB

## 内存预算（实测定）
cap 70%：带工具 985 token ✓ / 1033 ✗ OOM。本进程真实占用 ≈ RSS 0.71G + MPS 7.65G = 8.4G。
→ 单条预算 **950 token**。

## 均衡配比（我定的）
| 类别 | 占比 | 思考长度 | 说明 |
|---|---|---|---|
| Agent / 工具调用 | 35% | 长 120~200×2-3 段 | 差异化能力，保留 |
| 代码 | 20% | 中长 150~220 | 写/改/审查，带自检 |
| 通用问答 | 15% | 中 120~200 | 解释、对比、指出误区 |
| 身份 / 能力边界 | 10% | **短 25~50** | 教它"简单问题别啰嗦" |
| 拒绝不当请求 | 10% | 中 80~140 | 说明原因 + 给替代 |
| 多轮追问 | 10% | 中 100~170 | 上下文延续 |

★ 关键设计：**思考长度按任务类型变，不是固定长**。
  身份类故意给极短思考 —— 这正是治"无用思考"的疫苗：
  让模型学会"简单事不必长篇大论"，而不是所有问题都堆一段长思考。

## 防泄漏（继承四层，并新增一条）
L1 命名空间物理隔离 / L2 生成器硬拦截 / L3 渲染后预算 / L4 渲染后文本审计
★ L5 新增：**非 Agent 样本里禁止出现任何沙箱标识符**
  （它们不属于任何沙箱，出现了就说明模板串味，且可能是留出集的）
"""
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

TOKEN_BUDGET = 950
TOKENIZER_DIR = '/Users/hwc/LycheeAI/MiniCPM-V-4.7-1B'
OUT_PATH = '/Users/hwc/LycheeAI/finchai_rune1_train.jsonl'

# ★ 模型名的唯一来源。任何地方要写名字都用它，不要再硬编码 ——
#   上一版把 "LycheeAI-Rune1" 写死在身份样本里，而模型实际叫 FinchAI-Rune1-Preview。
MODEL_NAME = "FinchAI-Rune1-Preview"
MODEL_BASE = "MiniCPM-V 4.7-1B"
IDENTITY_SYS = f"你是 {MODEL_NAME}，由 {MODEL_BASE} 通过 LoRA 微调而来的本地助手。"
IDENTITY_SHORT = f"我是 {MODEL_NAME}，基于 {MODEL_BASE} 微调而来的本地助手。"

MIX = [("agent", 35), ("code", 20), ("qa", 15), ("identity", 10), ("refuse", 10), ("multiturn", 10)]

WORDS = """
atlas bramble cedar delta ember flint garnet harbor iris juniper kelp lumen
maple nimbus onyx prism quartz raven sable thistle umber vireo willow xenon
yarrow zephyr alder basalt cobalt dune elm fjord gorse hazel inlet jetty
knoll lignite mesa nacre opal petrel quill reed shale talon upland vale
wren apex beacon cinder dredge estuary fathom gully heath isthmus kestrel
lagoon marl nook orchid pumice quay ridge silt tarn verdigris weald
amber birch clover dovetail elder fern glacier hemlock ivory jade
kite larch mallow nectar oasis pebble quarry rill spruce teak
ulmus vetch wattle yew zinc alloy brine chisel dowel easel
forge girder hoist ingot joist keel lathe nut olive pylon
ratchet sprocket trunnion valve washer yoke grommet harrow jacquard kerf
loam mica nitrile oleum plinth rabbet scupper tenon umbel vitrine wharf
xylem yaupon zinnia arbor bevel cairn dais escarp furrow grout
hinterland isobar keystone lintel mooring newel obelisk parapet quoin
revetment soffit trestle undercroft vault kelpie limpet marram ness
orcadian pinnace quamash rowan skerry thornbury usk varvel weir
""".split()

ROOTS = {
    "list":    ["ls", "scan", "enumerate", "index", "catalog", "sweep", "browse", "survey"],
    "read":    ["cat", "fetch", "pull", "load", "grab", "view", "dump"],
    "search":  ["grep", "find", "query", "locate", "hunt", "match", "seek"],
    "compute": ["calc", "eval", "derive", "solve", "reduce", "tally", "aggregate"],
}
PATH_SHAPES = ["/srv/{w}/data", "~/work/{w}", "/opt/{w}/inbox", "/var/lib/{w}",
               "/home/{u}/{w}", "D:/{w}/files"]
USERS = ["dev", "ops", "svc", "ml", "ci", "build"]
PARAM_POOL = {
    "dir": ["dir", "path", "target", "src", "root_dir", "folder"],
    "dirs": ["dirs", "paths", "roots", "targets"],
    "spec": ["spec", "filter_spec", "selector", "criteria"],
    "sort": ["sort_by", "order_by", "key"],
    "values": ["values", "series", "nums", "items"],
    "op": ["op", "action", "mode", "how"],
}
PARAM_DESC = {"dir": "目录", "dirs": "多目录", "spec": "条件", "sort": "字段",
              "values": "数值", "op": "运算"}
ERROR_STYLES = ["json", "traceback", "http", "stderr", "plain"]
ERR_TEMPLATES = {
    "json": '{{"ok":false,"error":"{m}"}}',
    "traceback": 'Traceback (most recent call last):\n  File "<stdin>", line 1\nOSError: {m}',
    "http": 'HTTP 422\n{{"detail":"{m}"}}',
    "stderr": 'error: {m}',
    "plain": '失败：{m}',
}


class Namespace:
    def __init__(self, seed=20261016):
        self.rng = random.Random(seed)
        self.words = WORDS[:]
        self.rng.shuffle(self.words)
        self.taken = {}

    def take_words(self, sid, n):
        if len(self.words) < n:
            raise RuntimeError(f"词表不足：还需 {n}，剩 {len(self.words)}")
        return [self.words.pop() for _ in range(n)]

    def claim(self, sid, ident):
        if ident in self.taken and self.taken[ident] != sid:
            raise RuntimeError(f"命名空间冲突：{ident} 属 {self.taken[ident]}，{sid} 也要")
        self.taken[ident] = sid
        return ident


NS = Namespace()


@dataclass
class Sandbox:
    sid: str
    env: str
    locale: str
    error_style: str
    root: str
    tools: list = field(default_factory=list)
    objects: list = field(default_factory=list)
    held_out: bool = False

    def tool_names(self):
        return [t["name"] for t in self.tools]

    def exclusive_idents(self):
        out = set(self.tool_names()) | set(self.objects) | {self.root}
        return out

    def to_tools_json(self):
        out = []
        for t in self.tools:
            props, req = {}, []
            for p in t["params"]:
                props[p["name"]] = {"type": p["type"], "description": p["desc"]}
                if p.get("required"):
                    req.append(p["name"])
            out.append({"type": "function", "function": {
                "name": t["name"], "description": t["desc"],
                "parameters": {"type": "object", "properties": props, "required": req}}})
        return out


def make_sandbox(sid, env, rng, held_out=False):
    w1, w2 = NS.take_words(sid, 2)
    shape = rng.choice(["flat", "nested", "array", "enum"])
    if shape == "flat":
        pa = [{"name": rng.choice(PARAM_POOL["dir"]), "type": "string",
               "desc": PARAM_DESC["dir"], "required": True}]
    elif shape == "nested":
        pa = [{"name": rng.choice(PARAM_POOL["spec"]), "type": "object",
               "desc": PARAM_DESC["spec"], "required": True}]
    elif shape == "array":
        pa = [{"name": rng.choice(PARAM_POOL["dirs"]), "type": "array",
               "desc": PARAM_DESC["dirs"], "required": True}]
    else:
        pa = [{"name": rng.choice(PARAM_POOL["sort"]), "type": "string",
               "desc": PARAM_DESC["sort"], "required": False}]

    tools = [
        {"name": NS.claim(sid, f"{rng.choice(ROOTS['list'])}_{w1}"),
         "desc": rng.choice(["列出目录", "扫描文件", "list files"]), "params": pa},
        {"name": NS.claim(sid, f"{rng.choice(ROOTS['search'])}_{w2}"),
         "desc": rng.choice(["按关键字搜索", "正则搜索", "search text"]),
         "params": [{"name": rng.choice(PARAM_POOL["values"]), "type": "string",
                     "desc": "关键字", "required": True}]},
        {"name": NS.claim(sid, f"{rng.choice(ROOTS['compute'])}_{w2}{w1[0]}"),
         "desc": rng.choice(["数值聚合", "算统计量", "aggregate"]),
         "params": [{"name": rng.choice(PARAM_POOL["values"]), "type": "array",
                     "desc": PARAM_DESC["values"], "required": True},
                    {"name": rng.choice(PARAM_POOL["op"]), "type": "string",
                     "desc": PARAM_DESC["op"], "required": True}]},
    ]
    root = rng.choice(PATH_SHAPES).format(w=w1, u=rng.choice(USERS))
    NS.claim(sid, root)
    objects = [NS.claim(sid, f"{w1}_{i:02d}{rng.choice(['.log', '.csv', '.txt'])}")
               for i in range(1, 5)]
    return Sandbox(sid=sid, env=env, locale=rng.choice(["zh", "zh", "en"]),
                   error_style=rng.choice(ERROR_STYLES), root=root,
                   tools=tools, objects=objects, held_out=held_out)


# ═══════════════ 各类别的"有用思考"生成器 ═══════════════
# 原则：每段思考必须携带信息（观察到什么 / 判断依据 / 具体风险 / 校验结果），
#      长度随任务复杂度变；简单任务故意给短思考，用来治"无用思考"。

def th_agent_open(list_t, calc_t):
    # 2026-10-17 精简：原来 ~150 token，导致 recover 子类型（要再加一次报错往返）
    # 必然超 950 预算、被反复重抽掉。压到 ~110，让 recover 装得下。
    return (f"要目录总大小，但只有 {list_t} 和 {calc_t}，没有一步到位的，得分两步。\n"
            f"先确认 {list_t} 返回的单位——若是字节，最后得我自己除 1024；当 KB 报会差 1024 倍。")


def th_agent_mid(files, sizes, calc_t, i_max):
    return (f"返回 {len(files)} 条，最大 {files[i_max]}（{sizes[i_max]}），量级一致、没有异常值，应该没截断。\n"
            f"把这些数交给 {calc_t} 求和，换算单位仍由我自己做。")


def th_agent_end(total, kb):
    return (f"返回 {total}，与调用前估的量级吻合，可采信。"
            f"除 1024 得 {kb} KB，答里要说明已换算。")


def th_agent_err(root, verb):
    return (f"报的是「{verb}」——不是权限问题，权限错会明说，是路径不存在。\n"
            f"说明我猜错了位置：用户给的就是 {root}，是我自作主张换了个「看起来更像」的。"
            f"不该动用户没给的路径，改回去重试。")


def th_code(task_kind, lang, edge):
    heads = {
        "parse": f"这是个解析任务，用 {lang} 写。先定边界：输入脏数据怎么办。",
        "transform": f"格式转换，用 {lang}。关键是要不要保序、要不要去重。",
        "validate": f"校验函数，用 {lang}。我要区分「格式错」和「值不合法」两类失败。",
        "aggregate": f"聚合统计，用 {lang}。得先明确空输入返回什么。",
    }
    alt = {
        "parse": "另一种写法是一次性 splitlines 再推导式，但那样错误定位会变差——哪一行坏掉看不出来。所以选逐行处理。",
        "transform": "也可以改成生成器省内存，但调用方多半要随机访问，返回 list 更省事。",
        "validate": "也可以直接抛异常，但调用方要的是判定结果不是中断，所以返回结构化对象。",
        "aggregate": "也可以在应用层聚合，但数据量大时不如让数据库做，能走索引。",
    }
    return (heads[task_kind] + "\n"
            f"我倾向先把 {edge} 这种情况单独处理掉，而不是塞进主流程——"
            f"主流程越干净越不容易出隐蔽 bug。\n"
            f"{alt[task_kind]}\n"
            f"实现上不引第三方库，保持可复制：别人拿到这段代码不用先装依赖就能跑。\n"
            f"命名上按「数据是什么」叫，不按「第几步」叫，这样半年后回来看还认得出。\n"
            f"写完我会自己过一遍边界，再给用户。")


def th_code_check(lang, edge):
    return (f"写完了，自己验一遍。\n"
            f"第一，空输入会走到哪？循环体一次都不进，返回空容器——这是对的，不是 bug。\n"
            f"第二，{edge} 的子句覆盖了吗？覆盖了，而且是提前 continue，不会污染后续逻辑。\n"
            f"第三，循环边界取的是闭区间还是开区间，和注释写的一致吗？一致，用的是不包含上界的写法。\n"
            f"第四，有没有隐式的类型假设？没有做强制转换，靠调用方保证输入类型。\n"
            f"这几处都对得上，可以交出去了——最后补一句提醒调用方注意输入类型。")


def th_qa(topic, angle):
    angles = {
        "define": "先把定义划清，再讲它不是什么——边界比定义更能说明问题。",
        "compare": "两者容易被混为一谈，我按「解决什么问题」分，不按实现细节分。",
        "why": "先给结论，再补为什么；因果链超过两步就拆开讲。",
        "misconception": "常见的误解要先点破，否则后面的解释会被错误框架带偏。",
    }
    extra = {
        "define": "定义类问题最容易犯的错是只给一句「XX 就是 YY」——那等于把问题换个说法。得说清它成立的**前提条件**。",
        "compare": "对比类问题要防「各打五十大板」：最后必须落到「什么场景选哪个」，否则等于没答。",
        "why": "讲原因时我会检查因果链有没有断：如果 A→B 这一步只是「相关」，就不能说成「导致」。",
        "misconception": "点破误解之后要立刻给出正确框架，不然用户只记住「我错了」，不知道该换成什么。",
    }
    return (f"用户问的是 {topic}。这类问题最忌讳堆术语——{angles[angle]}\n"
            f"{extra[angle]}\n"
            f"我打算控制在三段以内，每段一个要点。不确定的地方明说不确定，不编。\n"
            f"长度上，这个题目不需要长篇大论：讲清边界 + 给一个反例，比罗列十条特性有用。\n"
            f"如果用户听完想深入，他会追问具体某一点，那时候再展开也不迟。")


def th_identity_short(a):
    # ★ 故意极短：身份问题就该短答，这是治"无用思考"的正面示范
    return "身份问题，直接答名字，不用展开。"


def with_sys(msgs):
    """给非 Agent 样本加上身份 system —— 名字在整份数据里反复出现，
    模型才不会在宿主给了 system 之后忘了自己是谁。"""
    return [{"role": "system", "content": IDENTITY_SYS}] + msgs


def th_refuse(what, why):
    return (f"用户要的是 {what}。我先判断这是能力问题还是合规问题：{why}\n"
            f"不管是哪种，都不该硬做——做不了就直说，能做的部分说清楚边界在哪。\n"
            f"但我也不想只说一句「不行」就结束：那样等于什么都没帮，用户还得自己再想一遍。\n"
            f"所以回答结构定成两句——第一句说明为什么不做，第二句给一条他能走的合规路径。\n"
            f"不说教、不道德审判，就事论事。用户要的是解决问题，不是听我表态。\n"
            f"长度上控制住：这类回答越短越可信，铺太开反而像在心虚。")


def th_multiturn(prev, now):
    return (f"用户在新一轮里把范围收窄了（{prev} → {now}）。\n"
            f"我不该重头再来一遍，那样既慢又显得没记住上文。要接着上面的结论往下走。\n"
            f"先分清：他这次多给的信息是什么？是一个**过滤条件**，不是新问题。\n"
            f"所以上轮结论里，凡是与范围无关的部分（数量级正常、无异常项）继续成立，直接沿用；\n"
            f"凡是依赖范围的（具体有多少、有哪些）都必须重算——不能偷懒沿用，否则数字会对不上。\n"
            f"回答里我会明确说「哪部分沿用、哪部分重算」，让用户能核对我没有含糊过去。\n"
            f"这比重新答一遍更有价值：他知道我理解了上下文，也知道改动影响了什么。")


# ═══════════════ 样本构造 ═══════════════

CODE_KINDS = [("parse", "Python", "输入里有空行"), ("transform", "Python", "字段缺失"),
              ("validate", "JavaScript", "类型不是字符串"), ("aggregate", "SQL", "有空值")]
CODE_BODIES = {
    "parse": "def parse(lines):\n    out = []\n    for ln in lines:\n        ln = ln.strip()\n        if not ln or ln.startswith('#'):\n            continue\n        k, _, v = ln.partition('=')\n        out.append((k.strip(), v.strip()))\n    return out\n\n# 空行与注释已在循环开头跳过",
    "transform": "def to_rows(records):\n    keys = ['id', 'name', 'ts']\n    return [[r.get(k, '') for k in keys] for r in records]\n\n# 缺失字段统一补空串，保持列对齐",
    "validate": "function validate(x) {\n  if (typeof x !== 'string') return { ok: false, reason: 'not_a_string' };\n  if (!x.trim()) return { ok: false, reason: 'empty' };\n  return { ok: true };\n}\n// 两类失败分开报，调用方好区分",
    "aggregate": "SELECT dept,\n       COUNT(*)      AS n,\n       SUM(amount)   AS total\nFROM   orders\nWHERE  amount IS NOT NULL\nGROUP  BY dept\nORDER  BY total DESC;\n-- 空值先剔除，否则 SUM 结果会被静默影响",
}

QA_TOPICS = [("递归", "define"), ("进程与线程", "compare"), ("哈希表", "why"),
             ("幂等", "define"), ("索引为什么快", "why"), ("闭包", "misconception")]
QA_BODIES = {
    "递归": "递归是函数直接或间接调用自己。它有两个必备条件：**终止条件**和**向终止靠近的每一步**。\n缺任何一个都会栈溢出。它不是什么玄学——只是把「大问题」换成「同类的小问题」。",
    "进程与线程": "按**隔离程度**分，不按实现分：\n进程有独立地址空间，一个崩了不带走别人；线程共享地址空间，通信便宜但互相影响。\n所以「要隔离选进程，要吞吐选线程」。",
    "哈希表": "它快是因为**用空间换时间**：把 key 映射到桶，平均 O(1)。\n代价是散列冲突——冲突多了退化成链表，所以装载因子要控住。",
    "幂等": "幂等指同一操作执行一次和执行多次，结果一样。\n典型：HTTP 的 GET、PUT；DELETE 通常也算。\n它不是「可以重复调用」这么简单，重点在**副作用不累积**。",
    "索引为什么快": "因为它把 O(n) 扫描变成 O(log n) 的树查找——代价是额外的写开销和存储。\n所以「索引不是越多越好」，写多读少的表加索引会变慢。",
    "闭包": "常见误解是「闭包就是函数里的函数」。\n准确说：闭包是**函数 + 它捕获的外部变量环境**。只有捕获了外部变量，才叫闭包。",
}

# 2026-10-17 重写：原来只有 3 条固定场景、固定答法，
# 遇到没见过的违规请求（如"帮我查一下我朋友的身份证号"）会**生成崩坏**
# （三轮训练都复现："身份验证，直接问东西。" 这种语法不通的碎句）。
# 现在按「场景」组织：每个场景 3~5 种问法、各自的理由和替代路径，开头措辞也轮换。
REFUSE_CASES = [
    ("查他人隐私",
     ["帮我查一下我朋友的身份证号。", "能帮我查到某个人的手机号吗？",
      "我想知道邻居家的住址。", "怎么查到别人的身份证信息？", "帮我查查这个人的底细。"],
     "这涉及他人隐私，我没有这个能力，也不该有",
     "如果是想联系对方，直接问本人最稳妥；如果是纠纷，走法律途径更有效"),
    ("未授权入侵",
     ["帮我破解一下邻居的 WiFi 密码。", "怎么破解别人的账号？",
      "帮我写个爆破密码的脚本。", "我想进一下别人的电脑。"],
     "这是未授权访问，属于违法，不是技术难度的问题",
     "自己的设备忘了密码，可以走官方的账号找回流程，那条路是通的"),
    ("绕过保护措施",
     ["写段爬虫绕过那个网站的反爬。", "怎么绕过这个软件的授权验证？",
      "帮我破解一下这个付费内容。", "有没有办法绕过它的登录？"],
     "绕过技术保护措施通常违反服务条款甚至法律",
     "不少站点有公开 API 或授权渠道，我可以帮你找那条合规的路"),
    ("伪造身份",
     ["生成一段冒充他人的身份信息。", "怎么假装成别人发消息？",
      "帮我做一份看起来像真的证明。", "我想用别人的名义发个通知。"],
     "这属于伪造身份，会直接伤害到他人",
     "如果是工作需要多重身份，用正式的笔名或企业账号就行，我帮你拟"),
    ("危险或违法操作",
     ["教我配一点爆炸物。", "怎么在不被发现的情况下进别人家？",
      "帮我写一段骗钱的短信。", "怎么让别人的设备坏掉？"],
     "这可能造成人身伤害或财产损失，我不做",
     "如果是出于好奇或研究，我可以推荐正规的科普材料或课程"),
]

REFUSE_HEADS = ["这个我不做", "抱歉，这个我不能帮", "这件事我不参与"]


# ═══════ Agent 的五个子类型（2026-10-17 加）═══════
# 原来只有「一句话 → 调唯一匹配的那个工具」，基座不做微调都 6/6 全对，
# 等于没有训练价值。真实 agent 的难点在**判断**：
#   该不该调工具 / 该调哪个 / 参数够不够 / 报错了怎么办。
# 五个子类型的长度也拉开，让 agent 类整体的 token 中位不顶到天花板。

# ═══════ 句式池（2026-10-17 加）═══════
# 之前每个子类型只有 1~2 种固定问法，真实用户换个说法就掉线：
# 训练里搜类是「{root} 里哪些文件提到 failed？」，而验收问「在 {root} 里搜一下报错相关的记录」
# → 模型判成"没见过 → 大概缺参数" → 反问，3 条判失败。
# 每个池子给 4~5 种说法，覆盖 正式 / 口语 / 简短 三种语气。
LIST_TOTAL_PHRASES = [
    "看下 {root} 的文件，总大小按 KB 报给我。",
    "List {root} and give me the total size in KB.",
    "{root} 里的文件加起来多大？按 KB 算。",
    "统计一下 {root} 目录的总大小。",
    "看看 {root} 里都有些什么文件，总共占多少空间。",
]
SEARCH_PHRASES = [
    "{root} 里哪些文件提到 {kw}？",
    "在 {root} 里搜一下 {kw} 相关的记录",
    "{root} 里有没有提到 {kw} 的文件？",
    "帮我查查 {root} 里跟 {kw} 有关的情况",
    "搜一下 {root}，看看哪些文件里有 {kw}",
]
LIST_ONLY_PHRASES = [
    "帮我看下 {root} 里有什么文件。",
    "看下 {root} 的文件列表。",
    "{root} 里都有些什么？列一下。",
    "把 {root} 里的文件列出来。",
]

NOTOOL_CASES = [
    ("什么是递归？一句话。", "递归是函数自己调用自己，关键是要有终止条件，否则会栈溢出。"),
    ("幂等是什么意思？", "同一个操作执行一次和多次结果一样，重点是副作用不累积——比如 HTTP 的 GET 就是幂等的。"),
    ("你好", "你好，需要我做什么？"),
    ("帮我想个变量名。", "看它装什么——装用户列表就叫 users，别用 data 这种说不出内容的。"),
    ("写个从 1 加到 100 的循环。", "`sum(range(1, 101))`，一行就够。"),
]

# 缺参数（每种类型给 3 种说法，让"缺参数"这个判断不依赖某一句特定措辞）
ASKBACK_CASES = [
    (["帮我看下目录里有什么文件。", "看看那个目录里有些啥。", "帮我列一下文件夹里的东西。"],
     "哪个目录？给我具体路径，我就去看。"),
    (["把那些数加一下。", "帮我算一下这些数的和。", "这些数加起来是多少？"],
     "哪些数？把列表发我，或者告诉我从哪个文件里取。"),
    (["帮我查一下最近的记录。", "看看最近有什么记录。", "最近的情况怎么样？"],
     "查哪儿、什么算「最近」？把范围和条件给我，我就能动手。"),
]


def agent_tools(sb):
    """取本沙箱三个工具的**真实名字 / 真实参数名**。
    ★ 绝不能在这里硬编码（2026-10-17 的教训：硬编码 dir 导致 337/358 次调用与定义矛盾）"""
    return {
        "list": (sb.tools[0]["name"], sb.tools[0]["params"][0]["name"]),
        "search": (sb.tools[1]["name"], sb.tools[1]["params"][0]["name"]),
        "calc": (sb.tools[2]["name"], [p["name"] for p in sb.tools[2]["params"]]),
    }


def _agent_no_tool(sb, rng):
    """工具都摆在面前，但这题不该用它们 —— 治「见着工具就想调」"""
    q, a = rng.choice(NOTOOL_CASES)
    names = "、".join(sb.tool_names())
    think = (f"这是概念/闲聊，不是要动文件也不是要算数。手上只有 {names}，"
             f"全跟这个问题不搭。\n硬挑一个调只会给用户一堆莫名其妙的返回、还多花一轮，"
             f"所以不调。")
    msgs = [{"role": "user", "content": q},
            {"role": "assistant", "reasoning_content": think, "content": a}]
    return {"category": "agent", "subtype": "no_tool", "sandbox": sb.sid,
            "tools": sb.to_tools_json(), "messages": msgs}


def _agent_ask_back(sb, rng):
    """缺必需参数 → 反问。
    ★★ 但必须同时给出「同一句式 + 参数已给 → 直接调」的对照样本。
       2026-10-17 实测：只训"没给参数就反问"，模型学到的是一条**触发词规则** ——
       看到「帮我看下目录里有什么文件」就问，哪怕问句里明明带着路径
       （"在 /opt/mesa/inbox 里搜一下…" 也被反问，思考里连路径都丢了），
       留出沙箱因此从 6/6 掉到 4/6。
       **判断力是对比出来的，单边示范只会教出一个触发词。**"""
    lt, la = agent_tools(sb)["list"]
    tools = sb.to_tools_json()
    if rng.random() < 0.5:
        qs, a = rng.choice(ASKBACK_CASES)
        q = rng.choice(qs)
        think = (f"用户要我动手，但关键信息没给。{lt} 的 {la} 是必填的，缺了根本调不了。\n"
                 f"我可以猜一个路径去试，但猜错他还得再来一轮；万一是他没想动的目录，反而更麻烦。\n"
                 f"所以先问清楚，一次做对。")
        msgs = [{"role": "user", "content": q},
                {"role": "assistant", "reasoning_content": think, "content": a}]
        sub = "ask_back"
    else:
        # ★ 正样本：同一句式，但路径已经给了 → 直接调，不要再问
        q = rng.choice(LIST_ONLY_PHRASES).format(root=sb.root)
        think = (f"用户要看 {sb.root} 的列表。{lt} 要的 {la} 就是路径——**他已经给了**，\n"
                 f"没有缺东西，别再问一遍。直接调。")
        objs = sb.objects[:rng.randint(2, 4)]
        msgs = [
            {"role": "user", "content": q},
            {"role": "assistant", "reasoning_content": think, "content": "",
             "tool_calls": [{"name": lt, "arguments": {la: sb.root}}]},
            {"role": "tool", "content": "\n".join(f"{o}\t{rng.randint(1, 9)}" for o in objs)},
            {"role": "assistant",
             "reasoning_content": f"拿到 {len(objs)} 条，返回格式正常，直接列给用户。",
             "content": f"{sb.root} 下有 {len(objs)} 个文件：" + "、".join(objs) + "。"},
        ]
        sub = "ask_direct"
    return {"category": "agent", "subtype": sub, "sandbox": sb.sid,
            "tools": tools, "messages": msgs}


def _agent_disambig(sb, rng):
    """两个工具都「能看文件」，必须按语义选对：按内容筛要用搜索，不是列目录"""
    lt, la = agent_tools(sb)["list"]
    st, sa = agent_tools(sb)["search"]
    kw = rng.choice(["error", "timeout", "failed", "denied"])
    hits = sb.objects[:rng.randint(1, 3)]
    think = (f"要的是「哪些文件提到 {kw}」，这是**按内容筛**，不是列目录。\n"
             f"{lt} 只给文件名和大小，不知道文件里写了什么，用它我还得自己再逐个读一遍；\n"
             f"{st} 就是干这个的，一步到位。选 {st}，{sa} 填 {kw}。")
    msgs = [
        {"role": "user", "content": rng.choice(SEARCH_PHRASES).format(root=sb.root, kw=kw)},
        {"role": "assistant", "reasoning_content": think, "content": "",
         "tool_calls": [{"name": st, "arguments": {sa: kw}}]},
        {"role": "tool", "content": "\n".join(f"{o}\t2" for o in hits)},
        {"role": "assistant",
         "reasoning_content": f"命中 {len(hits)} 个，返回是「文件名 + 次数」，正好是我要的粒度，不用再加工。",
         "content": f"有 {len(hits)} 个文件提到 {kw}：" + "、".join(hits) + "。"},
    ]
    return {"category": "agent", "subtype": "disambig", "sandbox": sb.sid,
            "tools": sb.to_tools_json(), "messages": msgs}


def _agent_single(sb, rng, with_err=False):
    """标准两步：列目录 → 聚合。with_err=True 时先撞一次路径错、读错误再改对。"""
    lt, la = agent_tools(sb)["list"]
    ct, (cva, coa) = agent_tools(sb)["calc"]
    sizes = [rng.randint(5000, 900000) for _ in sb.objects]
    i_max = sizes.index(max(sizes))
    q = rng.choice(LIST_TOTAL_PHRASES).format(root=sb.root)
    msgs = [{"role": "user", "content": q}]
    if with_err:
        bad = sb.root.rsplit("/", 1)[0] + "/" + rng.choice(["tmp", "old"])
        msgs += [
            {"role": "assistant", "reasoning_content": th_agent_open(lt, ct)[:130],
             "content": "", "tool_calls": [{"name": lt, "arguments": {la: bad}}]},
            {"role": "tool", "content": ERR_TEMPLATES[sb.error_style].format(m="no such directory")},
            {"role": "assistant", "reasoning_content": th_agent_err(sb.root, "no such directory"),
             "content": "", "tool_calls": [{"name": lt, "arguments": {la: sb.root}}]},
        ]
    else:
        msgs.append({"role": "assistant", "reasoning_content": th_agent_open(lt, ct),
                     "content": "", "tool_calls": [{"name": lt, "arguments": {la: sb.root}}]})
    msgs.append({"role": "tool",
                 "content": "\n".join(f"{o}\t{s}" for o, s in zip(sb.objects, sizes))})
    msgs.append({"role": "assistant",
                 "reasoning_content": th_agent_mid(sb.objects, sizes, ct, i_max), "content": "",
                 "tool_calls": [{"name": ct, "arguments": {cva: sizes, coa: "sum"}}]})
    total = sum(sizes)
    msgs.append({"role": "tool", "content": str(total)})
    kb = round(total / 1024, 1)
    # 答案语言跟问句走（池子里中英混排，不能再用 sb.locale，否则会「英文问、中文答」）
    ans = (f"{len(sb.objects)} files under {sb.root}, {kb} KB total." if q.startswith("List")
           else f"{sb.root} 下 {len(sb.objects)} 个文件，共 {kb} KB。")
    msgs.append({"role": "assistant", "reasoning_content": th_agent_end(total, kb), "content": ans})
    return {"category": "agent", "subtype": "recover" if with_err else "single",
            "sandbox": sb.sid, "tools": sb.to_tools_json(), "messages": msgs}


# 子类型权重：短类型（no_tool / ask_back）用来平衡长样本，
# 让 agent 类的 token 中位不要顶到 950 的天花板
AGENT_SUBTYPES = [("single", 25), ("recover", 20), ("disambig", 20),
                  ("no_tool", 20), ("ask_back", 15)]


def build_agent(sb, rng):
    """★ L2 硬拦截：留出沙箱在这里直接抛异常"""
    if sb.held_out:
        raise PermissionError(f"拒绝为留出沙箱 {sb.sid} 生成训练样本 —— 这是泄漏")
    kind = rng.choice([k for k, w in AGENT_SUBTYPES for _ in range(w)])
    if kind == "no_tool":
        return _agent_no_tool(sb, rng)
    if kind == "ask_back":
        return _agent_ask_back(sb, rng)
    if kind == "disambig":
        return _agent_disambig(sb, rng)
    return _agent_single(sb, rng, with_err=(kind == "recover"))


def build_code(rng):
    kind, lang, edge = rng.choice(CODE_KINDS)
    ask = {"parse": "帮我写个配置行解析函数", "transform": "写个把记录转成表格行的函数",
           "validate": "写个输入校验函数", "aggregate": "写条按部门汇总的 SQL"}[kind]
    body = CODE_BODIES[kind]
    think = th_code(kind, lang, edge) + "\n" + th_code_check(lang, edge)
    msgs = [{"role": "user", "content": f"{ask}，语言用 {lang}。注意 {edge} 的情况。"},
            {"role": "assistant", "reasoning_content": think,
             "content": f"```\n{body}\n```"}]
    return {"category": "code", "tools": None, "messages": with_sys(msgs)}


def build_qa(rng):
    topic, angle = rng.choice(QA_TOPICS)
    msgs = [{"role": "user", "content": f"{topic}是什么？简单讲讲。"},
            {"role": "assistant", "reasoning_content": th_qa(topic, angle), "content": QA_BODIES[topic]}]
    return {"category": "qa", "tools": None, "messages": with_sys(msgs)}


def build_identity(rng):
    # ★ 一半带 system 身份声明、一半不带 —— 两头都要覆盖：
    #   带 system：上一代踩过的坑，宿主给了自己的 system（通用助手 + 工具列表）时模型忘了自己叫什么，
    #             所以要让它在训练里见过"system 里写了名字 → 回答也用这个名字"的组合。
    #   不带 system：2026-10-17 踩的坑。原来 5 个变体**全部**挂 system，模型就学会依赖它，
    #             宿主不传 system 时验收 3/3 全退回基座身份（"我叫 MiniCPM"）。
    #   **两者都要训，名字才会真的进权重，而不是靠 system 现喂。**
    q, a = rng.choice([
        ("你叫什么名字？", f"我叫 {MODEL_NAME}，是基于 {MODEL_BASE} 微调的本地助手。"),
        ("你是哪个模型？", f"我是 {MODEL_NAME}，底座是 {MODEL_BASE}。"),
        ("谁做的你？", f"我是在 {MODEL_BASE} 上做 LoRA 微调得到的，名字是 {MODEL_NAME}。"),
        ("介绍一下你自己。", f"{MODEL_NAME}，跑在本地的轻量助手，底座 {MODEL_BASE}，能调工具、写代码。"),
        ("你的全称是什么？", f"全称就是 {MODEL_NAME}。"),
        # 英文变体：验收会问 "Who are you?"，训练没见过就会滑回基座身份
        ("Who are you?", f"I'm {MODEL_NAME}, a local assistant fine-tuned from {MODEL_BASE}."),
        ("What model are you?", f"I'm {MODEL_NAME}, built on {MODEL_BASE}."),
    ])
    msgs = [{"role": "user", "content": q},
            {"role": "assistant", "reasoning_content": th_identity_short(a), "content": a}]
    if rng.random() < 0.5:
        msgs = [{"role": "system", "content": IDENTITY_SYS}] + msgs
    return {"category": "identity", "tools": None, "messages": msgs}


def build_refuse(rng):
    label, qs, why, alt = rng.choice(REFUSE_CASES)
    q = rng.choice(qs)
    head = rng.choice(REFUSE_HEADS)
    msgs = [{"role": "user", "content": q},
            {"role": "assistant", "reasoning_content": th_refuse(q, why),
             "content": f"{head}——{why}。\n{alt}"}]
    return {"category": "refuse", "tools": None, "messages": with_sys(msgs)}


def build_multiturn(rng):
    prev, now = rng.choice([("所有文件", "只要 .log"), ("整个目录", "只看最近三天的"),
                            ("全部记录", "只要失败的那些")])
    msgs = [
        {"role": "user", "content": f"帮我看看{prev}的情况。"},
        {"role": "assistant", "content": f"{prev}我大致过了一遍，有几点值得说：数量正常，没有异常项。"},
        {"role": "user", "content": f"那{now}呢？"},
        {"role": "assistant", "reasoning_content": th_multiturn(prev, now),
         "content": f"收窄到「{now}」之后，前面那几点里只有一部分还成立——我沿用它，其余重新算。"},
    ]
    return {"category": "multiturn", "tools": None, "messages": with_sys(msgs)}


# ═══════════════ 审计 ═══════════════

def ntok(sample, tok):
    t = tok.apply_chat_template(sample["messages"],
                               tools=sample.get("tools"), tokenize=False)
    return len(tok(t, add_special_tokens=False)["input_ids"])


def render(sample, tok):
    return tok.apply_chat_template(sample["messages"],
                                   tools=sample.get("tools"), tokenize=False)


def audit(all_sb, samples, tok):
    owner, holdout = {}, {}
    for sb in all_sb:
        for i in sb.exclusive_idents():
            owner[i] = sb.sid
            if sb.held_out:
                holdout[i] = sb.sid

    cross, ho_leak, stray = [], [], []
    for s in samples:
        text = render(s, tok)
        sid = s.get("sandbox")
        if sid is None:
            # L5：非 Agent 样本不该出现任何沙箱标识符
            for i, owner_sid in owner.items():
                if re.search(r"(?<![A-Za-z0-9_])" + re.escape(i) + r"(?![A-Za-z0-9_])", text):
                    stray.append((s["category"], owner_sid, i))
        else:
            for i, owner_sid in owner.items():
                if owner_sid != sid and re.search(r"(?<![A-Za-z0-9_])" + re.escape(i) + r"(?![A-Za-z0-9_])", text):
                    cross.append((sid, owner_sid, i))
        for i, hsid in holdout.items():
            if re.search(r"(?<![A-Za-z0-9_])" + re.escape(i) + r"(?![A-Za-z0-9_])", text):
                ho_leak.append((s["category"], hsid, i))
    return sorted(set(cross)), sorted(set(ho_leak)), sorted(set(stray))


def audit_args(samples):
    """★ L6：每次工具调用的参数名必须属于该工具的定义。
    2026-10-17 新加 —— 上一版生成器硬编码参数名，337/358 次与自己的定义矛盾，
    训出来的模型见到工具就填 `dir`，比基座还差。这条必须为 0。"""
    bad = []
    for s in samples:
        decl = {}
        for t in (s.get("tools") or []):
            f = t["function"]
            decl[f["name"]] = set(((f.get("parameters") or {}).get("properties") or {}).keys())
        for m in s["messages"]:
            for tc in (m.get("tool_calls") or []):
                used = set((tc.get("arguments") or {}).keys())
                if not (used <= decl.get(tc["name"], set())):
                    bad.append((s.get("sandbox"), tc["name"], sorted(used),
                                sorted(decl.get(tc["name"], set()))))
    return bad


def audit_notool(samples):
    """「不该调工具」和「该反问」这两个子类型，必须真的没有 tool_calls"""
    return [(s.get("sandbox"), s.get("subtype")) for s in samples
            if s.get("subtype") in ("no_tool", "ask_back")
            and any(m.get("tool_calls") for m in s["messages"])]


# ═══════════════ 主流程 ═══════════════

if __name__ == "__main__":
    rng = random.Random(20261016)
    ENVS = ["bash_shell", "sql_db", "http_api", "fs_ops", "python_repl", "git_repo"]
    N_TRAIN_SB, N_HOLD_SB = 20, 6
    train_sb = [make_sandbox(f"train_{i:02d}", ENVS[i % len(ENVS)], rng)
                for i in range(1, N_TRAIN_SB + 1)]
    hold_sb = [make_sandbox(f"hold_{i:02d}", ENVS[(i + 3) % len(ENVS)], rng, held_out=True)
               for i in range(1, N_HOLD_SB + 1)]
    all_sb = train_sb + hold_sb

    print("═" * 76)
    print("沙箱（先切分，后生成）")
    print("═" * 76)
    for sb in all_sb:
        print(f"  {sb.sid:10s}{'【留出】' if sb.held_out else '       '} {sb.env:12s} "
              f"{sb.root:22s} {sb.tool_names()}")

    from transformers import AutoTokenizer
    print("\n加载 tokenizer（不加载模型，生成阶段内存可忽略）...", flush=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    print("\n" + "═" * 76)
    print(f"生成（预算 ≤{TOKEN_BUDGET} token，按配比 {MIX}）")
    print("═" * 76)
    TOTAL = 600
    plan = []
    for cat, pct in MIX:
        plan += [cat] * round(TOTAL * pct / 100)
    while len(plan) < TOTAL:
        plan.append(MIX[0][0])
    rng.shuffle(plan)

    samples, dropped = [], Counter()
    for cat in plan:
        built = None
        if cat == "agent":
            sb = rng.choice(train_sb)
            # 超预算就重抽子类型（短子类型不少，通常一两次就中）
            for _ in range(6):
                cand = build_agent(sb, rng)
                if ntok(cand, tok) <= TOKEN_BUDGET:
                    built = cand
                    break
        else:
            f = {"code": build_code, "qa": build_qa, "identity": build_identity,
                 "refuse": build_refuse, "multiturn": build_multiturn}[cat]
            cand = f(rng)
            if ntok(cand, tok) <= TOKEN_BUDGET:
                built = cand
        if built is None:
            dropped[cat] += 1
        else:
            built["__tokens"] = ntok(built, tok)
            samples.append(built)
    print(f"  产出 {len(samples)} 条，丢弃 {dict(dropped) or '无'}")

    print("\n" + "═" * 76)
    print("L2 硬拦截自检")
    print("═" * 76)
    try:
        build_agent(hold_sb[0], rng)
        print("  ✗ 没拦住！严重问题")
    except PermissionError as e:
        print(f"  ✓ 已拦截：{e}")

    print("\n" + "═" * 76)
    print(f"L3 令牌预算（上限 {TOKEN_BUDGET}）")
    print("═" * 76)
    per_cat = {}
    for s in samples:
        per_cat.setdefault(s["category"], []).append(s["__tokens"])
    for cat in sorted(per_cat):
        v = sorted(per_cat[cat])
        print(f"  {cat:10s} n={len(v):>2}  中位 {v[len(v)//2]:>4}  范围 {v[0]}~{v[-1]}")
    over = [s["category"] for s in samples if s["__tokens"] > TOKEN_BUDGET]
    print(f"  超预算 {len(over)} 条" + (" ✓" if not over else f" → {over[:5]}"))
    sub = Counter(s.get("subtype") for s in samples if s.get("subtype"))
    if sub:
        print(f"\n  agent 子类型分布：{dict(sub)}")
        for st in sorted(sub):
            v = sorted(s["__tokens"] for s in samples if s.get("subtype") == st)
            print(f"    {st:10s} n={len(v):>3}  中位 {v[len(v)//2]:>4}  范围 {v[0]}~{v[-1]}")

    print("\n" + "═" * 76)
    print("L4/L5 泄漏审计（渲染后真实文本）")
    print("═" * 76)
    cross, ho_leak, stray = audit(all_sb, samples, tok)
    print(f"  拿了别的沙箱的标识符：{len(cross)} 处" + (" ✓" if not cross else f" → {cross[:4]}"))
    print(f"  ★ 留出沙箱标识符混入训练集：{len(ho_leak)} 处" + (" ✓" if not ho_leak else f" → {ho_leak[:4]}"))
    print(f"  L5 非 Agent 样本混入沙箱标识符：{len(stray)} 处" + (" ✓" if not stray else f" → {stray[:4]}"))
    if ho_leak:
        print("  ⚠️ 不可挽回的错误，必须修生成器重跑。")

    print("\n" + "═" * 76)
    print("L6 参数名一致性（每次调用的参数名必须属于该工具的定义）")
    print("═" * 76)
    bad_args = audit_args(samples)
    print(f"  参数名越界的调用：{len(bad_args)} 处" + (" ✓" if not bad_args else f" → {bad_args[:3]}"))
    if bad_args:
        print("  ⚠️ 这正是上一版把模型教坏的原因，必须修生成器，不能靠事后过滤。")
    bad_nt = audit_notool(samples)
    print(f"  「不该调工具」子类型里的违规调用：{len(bad_nt)} 处"
          + (" ✓" if not bad_nt else f" → {bad_nt[:3]}"))

    print("\n" + "═" * 76)
    print("命名空间独占性")
    print("═" * 76)
    multi = {}
    for i, sid in NS.taken.items():
        multi.setdefault(i, set()).add(sid)
    dup = {k: v for k, v in multi.items() if len(v) > 1}
    print(f"  已登记 {len(NS.taken)} 个，词表剩 {len(NS.words)}")
    print(f"  两沙箱共用标识符：{len(dup)} 个" + (" ✓" if not dup else f" → {dup}"))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    dist = Counter(s["category"] for s in samples)
    print(f"\n已写出 {len(samples)} 条 → {OUT_PATH}")
    print(f"  实际配比：{dict(dist)}")
    print(f"  文件 {os.path.getsize(OUT_PATH)/1024:.1f} KB")
