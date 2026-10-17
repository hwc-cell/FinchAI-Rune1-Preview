#!/usr/bin/env python
"""FinchAI-Rune1-Preview 本地对话 CLI

用法:
    python rune1_chat.py            # 交互对话（默认带工具）
    python rune1_chat.py --demo     # 跑几个预设问题，不做交互
    python rune1_chat.py --no-tools # 纯对话，不给工具

交互命令:
    :think   切换「是否显示思考」
    :tools   切换「是否提供工具」
    :clear   清空上下文
    :quit    退出
"""
import os
import re
import sys
import ast
import math
import fnmatch
import argparse

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

MODEL_DIR = '/Users/hwc/LycheeAI/FinchAI-Rune1-Preview-f16'
MAX_NEW = 768
# 默认输出上限。依据：训练时序列 ≤950 token（含输入），输出太长会开始 OOD
# （表现为复读、跑飞）。用 --max-tokens 可覆盖。

SYSTEM = ("你是 FinchAI-Rune1-Preview，一个跑在本地的助手。回答简洁直接。\n"
          "调用工具时，路径参数必须**逐字符复制**用户给的路径，不要改写、缩写或替换。")

# ── 工具定义（真实可执行，不是演示桩）────────────────────────────
TOOLS = [
    {"type": "function", "function": {
        "name": "list_files",
        "description": "列出某个目录下的文件和子目录",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "目录路径"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "search_files",
        "description": "按关键词搜索文件名和文件内容（不支持 *.gguf 这类通配符）",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "关键词"},
                                      "path": {"type": "string", "description": "搜索范围目录"}},
                       "required": ["query", "path"]}}},
    {"type": "function", "function": {
        "name": "count_files",
        "description": "统计目录下匹配某通配模式的文件数量和总大小。要数个数时用这个，不要用 list_files 自己数",
        "parameters": {"type": "object",
                       "properties": {"pattern": {"type": "string",
                                                  "description": "文件名通配模式，如 *.gguf、*.log"},
                                      "path": {"type": "string", "description": "要统计的目录"}},
                       "required": ["pattern", "path"]}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "计算一个数学表达式",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string", "description": "表达式，如 789*123"}},
                       "required": ["expression"]}}},
]

# 一次给太多工具，1B 模型会选乱。按需裁剪：
#   files   = 列目录 / 数文件 / 计算（默认）
#   minimal = 只统计 + 计算
#   all     = 加上按内容搜索
TOOL_PRESETS = {
    'files':   [t for t in TOOLS if t['function']['name'] != 'search_files'],
    'minimal': [t for t in TOOLS if t['function']['name'] in ('count_files', 'calculate')],
    'all':     TOOLS,
}

BLOCKED = ('/System', '/private/var/db', '/dev', '/Library/Keychains')


def _safe_path(p):
    p = os.path.expanduser(str(p).strip())
    if not os.path.isabs(p):
        p = os.path.abspath(p)
    if any(p.startswith(b) for b in BLOCKED):
        return None
    return p


def tool_list_files(path):
    p = _safe_path(path)
    if not p:
        return "错误：不允许访问这个路径"
    if not os.path.isdir(p):
        return f"错误：{path} 不是目录或不存在"
    try:
        names = sorted(os.listdir(p))[:60]
    except PermissionError:
        return f"错误：没有权限读取 {path}"
    rows = []
    for n in names:
        full = os.path.join(p, n)
        size = ""
        if os.path.isfile(full):
            try:
                size = f"\t{os.path.getsize(full)}"
            except OSError:
                pass
        rows.append(f"{n}{'/' if os.path.isdir(full) else ''}{size}")
    return "\n".join(rows) if rows else "(空目录)"


def tool_search_files(query, path):
    p = _safe_path(path)
    if not p or not os.path.isdir(p):
        return f"错误：{path} 不是有效目录"
    q = str(query).lower()
    hits = []
    for root, dirs, files in os.walk(p):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for fn in files:
            full = os.path.join(root, fn)
            if q in fn.lower():
                hits.append(full)
                continue
            if len(hits) >= 20:
                break
            try:
                if os.path.getsize(full) > 2 * 1024 * 1024:
                    continue
                with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                    if q in f.read().lower():
                        hits.append(full)
            except Exception:
                pass
        if len(hits) >= 20:
            break
    return "\n".join(hits) if hits else f"没有找到包含「{query}」的文件"


_ALLOWED = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
_ALLOWED.update({'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum})


def tool_count_files(pattern, path):
    """★ 专治「让模型自己从列表里数」——它数不准会编数字"""
    p = _safe_path(path)
    if not p or not os.path.isdir(p):
        return f"错误：{path} 不是有效目录"
    pat = str(pattern).strip().lower()
    n, total = 0, 0
    for root, dirs, files in os.walk(p):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for fn in files:
            if fnmatch.fnmatch(fn.lower(), pat):
                n += 1
                try:
                    total += os.path.getsize(os.path.join(root, fn))
                except OSError:
                    pass
    return (f"匹配 {pattern} 的文件共 {n} 个，合计 {total:,} 字节"
            f"（{total / 1048576:.2f} MB）")


def tool_calculate(expression):
    expr = str(expression).strip()
    try:
        node = ast.parse(expr, mode='eval')
    except SyntaxError:
        return f"错误：表达式不合法（{expr}）"
    for n in ast.walk(node):
        if isinstance(n, (ast.Call,)):
            if not isinstance(n.func, ast.Name) or n.func.id not in _ALLOWED:
                return "错误：只支持四则运算和常用数学函数"
        elif isinstance(n, ast.Name) and n.id not in _ALLOWED:
            return f"错误：未知标识符 {n.id}"
    try:
        return str(eval(compile(node, '<calc>', 'eval'), {'__builtins__': {}}, _ALLOWED))
    except Exception as e:
        return f"错误：{type(e).__name__}: {e}"


REGISTRY = {
    'list_files': tool_list_files,
    'search_files': tool_search_files,
    'count_files': tool_count_files,
    'calculate': tool_calculate,
}

# ── 输出解析 ──────────────────────────────────────────────
CALL_RE = re.compile(r'<function=([A-Za-z_]\w*)>(.*?)</function>', re.S)
PARAM_RE = re.compile(r'<parameter=([A-Za-z_]\w*)>(.*?)</parameter>', re.S)


def split_think(text):
    """模型的思考以 '</think>' 收尾；没闭合就整段当正文"""
    if '</think>' in text:
        think, body = text.split('</think>', 1)
        return think.replace('<think>', '').strip(), body.strip()
    return '', text.strip()


def parse_calls(body):
    out = []
    for name, inner in CALL_RE.findall(body):
        args = {k: v.strip() for k, v in PARAM_RE.findall(inner)}
        out.append({'name': name, 'arguments': args})
    return out


def strip_tags(text):
    return CALL_RE.sub('', text).replace('</think>', '').strip()


def retry_with_user_path(name, args, result, user_text):
    """★ 模型偶尔会改写用户给的路径（训练时路径随机化的副作用）。
    工具报「不是有效目录」时，用用户原话里真实存在的路径自动重试一次。"""
    if ('不是有效目录' not in result and '不是目录或不存在' not in result
            and '不存在' not in result):
        return result, False, args
    pk = [k for k in args if 'path' in k.lower()]
    if not pk or not user_text:
        return result, False, args
    cands = re.findall(r'/(?:Users|home|var|opt|tmp|etc)[\w./\-]*', user_text)
    for c in sorted(set(cands), key=len, reverse=True):
        if os.path.isdir(c):
            new_args = {**args, pk[0]: c}
            try:
                return REGISTRY[name](**new_args), True, new_args
            except Exception:
                continue
    return result, False, args


# ── 模型 ──────────────────────────────────────────────────
def load():
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print('加载模型…', end='', flush=True)
    t0 = __import__('time').time()
    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, low_cpu_mem_usage=True).to('mps')
    model.eval()
    print(f' 完成（{__import__("time").time() - t0:.1f}s，'
          f'{torch.mps.current_allocated_memory()/2**30:.2f} GB）')
    return proc, model


def generate(model, tok, messages, tools=None, temperature=0.3, max_new=None):
    import torch

    max_new = max_new or MAX_NEW
    msgs = [{'role': 'system', 'content': SYSTEM}] + messages
    text = tok.apply_chat_template(
        msgs, tools=tools, tokenize=False, add_generation_prompt=True)
    enc = tok(text=[text], return_tensors='pt').to('mps')
    n_in = enc['input_ids'].shape[-1]
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tok.tokenizer.eos_token_id)

    gen = out[0][n_in:]
    raw = tok.tokenizer.decode(gen, skip_special_tokens=True)
    # 没遇到 eos 就停 = 撞到上限，回答被截断
    hit_limit = len(gen) >= max_new and gen[-1].item() != tok.tokenizer.eos_token_id
    return raw, hit_limit


def turn(model, tok, messages, tools, show_think, max_new=None):
    """跑一轮（可能含多次工具调用），返回最终回答文本"""
    import time

    # 模板要求：最后一条必须是 user（或 tool），否则会抛
    # "No user query found in messages"
    if not messages or messages[-1]['role'] not in ('user', 'tool'):
        raise ValueError('turn() 需要 messages 最后一条是 user 或 tool')
    # 用户原话（用于工具路径纠错）
    user_text = ''
    for m in reversed(messages):
        if m['role'] == 'user':
            user_text = m['content']
            break
    for _ in range(4):
        t0 = time.time()
        raw, hit_limit = generate(model, tok, messages, tools, max_new=max_new)
        dt = time.time() - t0

        think, body = split_think(raw)
        if think and show_think:
            print(f'\n\033[2m〔思考〕{think}\033[0m')
        if hit_limit:
            print(f'\033[33m  ⚠ 输出达到上限 {max_new or MAX_NEW} token，回答可能被截断'
                  f'（用 --max-tokens 提高）\033[0m')

        calls = parse_calls(body)
        if not calls:
            answer = strip_tags(body)
            if not answer:
                answer = '（这一轮没有输出，可能是判断不需要调用工具）'
            print(f'\n{answer}')
            print(f'\033[2m  {dt:.1f}s\033[0m' if show_think else '')
            messages.append({'role': 'assistant', 'content': answer,
                             'reasoning_content': think})
            return answer

        messages.append({'role': 'assistant', 'content': strip_tags(body),
                         'reasoning_content': think})
        for c in calls:
            name, args = c['name'], c['arguments']
            if name not in REGISTRY:
                result = f'错误：没有名为 {name} 的工具'
            else:
                try:
                    result = REGISTRY[name](**args)
                    result, fixed, args_used = retry_with_user_path(
                        name, args, result, user_text)
                    if fixed:
                        args = args_used
                        if show_think:
                            print('\033[33m    （路径已自动纠正为用户给的路径）\033[0m')
                except TypeError as e:
                    result = f'错误：参数不对（{e}）'
                except Exception as e:
                    result = f'错误：{type(e).__name__}: {e}'
            if show_think:
                shown = ', '.join(f'{k}={v!r}' for k, v in args.items())
                print(f'\n\033[36m  → 调用 {name}({shown})\033[0m')
                preview = result if len(result) < 220 else result[:220] + ' …'
                print('\033[2m' + '\n'.join('    ' + l for l in preview.splitlines()[:8]) + '\033[0m')
            messages.append({'role': 'tool', 'content': result})
    return ''


DEMO = [
    ('身份（不带 system 也能答对）', '你叫什么名字？'),
    ('判断力：不该调工具', '幂等是什么意思？'),
    ('判断力：缺参数要反问', '帮我看下目录里有什么文件。'),
    ('真实工具调用', '看下 /Users/hwc/LycheeAI 里有哪些 .gguf 文件，并告诉我总共几个'),
    ('真实计算', '算一下 1234*5678 等于多少'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--no-tools', action='store_true')
    ap.add_argument('--no-think', action='store_true')
    ap.add_argument('--max-tokens', type=int, default=MAX_NEW,
                    help=f'单轮最大输出 token（默认 {MAX_NEW}）')
    ap.add_argument('--toolset', choices=list(TOOL_PRESETS), default='files',
                    help='工具集：files(默认) / minimal / all')
    args = ap.parse_args()

    tok, model = load()
    show_think = not args.no_think
    max_new = args.max_tokens

    # :tools 在 all → files → minimal → 无 之间循环
    ORDER = ['all', 'files', 'minimal', None]
    idx = ORDER.index(args.toolset) if args.toolset in ORDER else 1
    if args.no_tools:
        idx = 3

    def current_tools():
        name = ORDER[idx]
        return TOOL_PRESETS[name] if name else None

    def ts_label():
        name = ORDER[idx]
        return name or '无工具'

    if args.demo:
        messages = []
        for title, q in DEMO:
            print(f'\n\033[1m── {title}\033[0m')
            print(f'问：{q}')
            messages.append({'role': 'user', 'content': q})
            turn(model, tok, messages, current_tools(), show_think, max_new)
        return

    print(f'\n输出上限 {max_new} token｜工具集 {ts_label()}。'
          f'命令：:think  :tools  :clear  :quit')
    messages = []
    while True:
        try:
            q = input('\n\033[1m你 > \033[0m').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in (':quit', ':q', 'exit'):
            break
        if q == ':clear':
            messages = []
            print('上下文已清空')
            continue
        if q == ':think':
            show_think = not show_think
            print(f'显示思考：{"开" if show_think else "关"}')
            continue
        if q == ':tools':
            idx = (idx + 1) % len(ORDER)
            names = [t['function']['name'] for t in (current_tools() or [])]
            print(f'工具集：{ts_label()}' + (f'（{", ".join(names)}）' if names else ''))
            continue
        messages.append({'role': 'user', 'content': q})
        turn(model, tok, messages, current_tools(), show_think, max_new)


if __name__ == '__main__':
    main()
