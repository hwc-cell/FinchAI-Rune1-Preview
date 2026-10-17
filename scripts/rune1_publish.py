"""FinchAI-Rune1-Preview 发布脚本：把 f16 完整版推到 HuggingFace / ModelScope。

用法: python rune1_publish.py hf|ms
注意: 必须 unset 所有代理变量（HF 需要直连，MS 必须关代理）。
"""
import os
import sys
import time

for _k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(_k, None)
os.environ['HF_HUB_DISABLE_XET'] = '1'

SRC = '/Users/hwc/LycheeAI/FinchAI-Rune1-Preview-f16'
REPO = 'FinchAI-Rune1-Preview-f16'
OWNER = 'whcl412'
MSG = 'FinchAI-Rune1-Preview：f16 完整版权重（含视觉）+ 模型卡'

platform = sys.argv[1].lower()
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
    from modelscope.hub.api import HubApi
    api = HubApi()
    try:
        api.create_model(model_id=f'{OWNER}/{REPO}')
        print('MS 仓库已创建', flush=True)
    except Exception as e:
        print(f'  create: {type(e).__name__}: {str(e)[:120]}（已存在则正常）', flush=True)
    for attempt in range(1, 4):
        try:
            api.upload_folder(repo_id=f'{OWNER}/{REPO}', folder_path=SRC, repo_type='model')
            print(f'MS 上传完成，用时 {time.time()-t0:.0f}s', flush=True)
            break
        except Exception as e:
            print(f'  upload 重试 {attempt}: {type(e).__name__}: {str(e)[:140]}', flush=True)
            time.sleep(8)
    else:
        print('MS 失败', flush=True)
        sys.exit(1)
