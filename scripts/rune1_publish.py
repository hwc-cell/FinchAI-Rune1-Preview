"""FinchAI-Rune1-Preview 发布脚本：把 f16 完整版推到 HuggingFace / ModelScope。

用法:
  python rune1_publish.py hf|ms [源码目录] [仓库名]

注意: 脚本自己会清掉所有代理变量（HF 直连可用；MS 必须关代理）。
"""
import os
import sys
import time

for _k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(_k, None)
os.environ['HF_HUB_DISABLE_XET'] = '1'

platform = sys.argv[1].lower()
SRC = sys.argv[2] if len(sys.argv) > 2 else '/Users/hwc/LycheeAI/FinchAI-Rune1-Preview-v9-f16'
REPO = sys.argv[3] if len(sys.argv) > 3 else 'FinchAI-Rune1-Preview-v2-f16'
OWNER = 'whcl412'
MSG = 'FinchAI-Rune1-Preview v2：f16 完整版权重（含视觉）+ 模型卡'
print(f'  源 {SRC}\n  目标 {OWNER}/{REPO}', flush=True)

t0 = time.time()

if platform == 'hf':
    from huggingface_hub import HfApi
    api = HfApi()
    for attempt in range(1, 4):
        try:
            api.create_repo(repo_id=f'{OWNER}/{REPO}', repo_type='model', exist_ok=True)
            print('HF 仓库就绪', flush=True)
            break
        except Exception as e:
            print(f'  create 重试 {attempt}: {type(e).__name__}: {str(e)[:100]}', flush=True)
            time.sleep(5)
    for attempt in range(1, 4):
        try:
            api.upload_folder(repo_id=f'{OWNER}/{REPO}', folder_path=SRC,
                              repo_type='model', commit_message=MSG)
            print(f'HF 上传完成，用时 {time.time()-t0:.0f}s', flush=True)
            break
        except Exception as e:
            print(f'  upload 重试 {attempt}: {type(e).__name__}: {str(e)[:140]}', flush=True)
            time.sleep(8)
    else:
        print('HF 失败', flush=True)
        sys.exit(1)

else:
    # ★ ModelScope 的本地 cookie 有 30 天过期设定，过期后新版 SDK 会直接判定「未登录」
    #   （报 10030000001 user not logged in），但**服务端会话往往仍然有效**。
    #   解决：从 cookies 里读出 m_session_id，显式传给 API，绕过客户端的过期检查。
    import pickle
    from modelscope.hub.api import HubApi

    _jar = pickle.load(open(os.path.expanduser('~/.modelscope/credentials/cookies'), 'rb'))
    TOK = next(c.value for c in _jar if c.name == 'm_session_id')
    print(f'  使用本地会话 token: {TOK[:12]}…', flush=True)

    api = HubApi()
    try:
        api.create_model(model_id=f'{OWNER}/{REPO}', token=TOK)
        print('MS 仓库已创建', flush=True)
    except Exception as e:
        print(f'  create: {type(e).__name__}: {str(e)[:110]}（已存在则正常）', flush=True)

    for attempt in range(1, 4):
        try:
            api.upload_folder(repo_id=f'{OWNER}/{REPO}', folder_path=SRC,
                              repo_type='model', token=TOK)
            print(f'MS 上传完成，用时 {time.time()-t0:.0f}s', flush=True)
            break
        except Exception as e:
            print(f'  upload 重试 {attempt}: {type(e).__name__}: {str(e)[:140]}', flush=True)
            time.sleep(8)
    else:
        print('MS 失败', flush=True)
        sys.exit(1)
