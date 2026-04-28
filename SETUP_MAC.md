# Mac (Apple M5 Pro, 24 GB) 使用指南

## 已完成的环境配置

- 安装了 `uv` (在 `~/.local/bin/uv`)
- Python 3.11.15 venv: `./.venv/`
- 安装依赖: `requirements-mac.txt`（PyTorch 2.11 自带 MPS 支持，已验证 `torch.backends.mps.is_available() == True`）
- 研究主题文件: `ai_scientist/ideas/handwritten_archive_ocr.md`

## 每次进入项目

```bash
cd ~/Desktop/mydocuments/codes/autoAi2/AI-Scientist-v2
export PATH="$HOME/.local/bin:$PATH"      # 让 uv 可用
source .venv/bin/activate
```

## 配置 API Keys（已配置）

API key 存放在项目根 `.env`（已被 `.gitignore` 忽略，不会进 git）。

每次新 shell 进入项目后：
```bash
source .venv/bin/activate
source load_env.sh           # 自动 export OPENAI_API_KEY 和 OPENAI_BASE_URL
```

当前 `.env` 内容（结构）：
```
OPENAI_API_KEY=sk-...                              # 已填
OPENAI_BASE_URL=https://gmn.chuangzuoli.com/v1     # 第三方 OpenAI 兼容代理
# S2_API_KEY=                                      # 可选，未配置
```

代理已验证：
- ✅ `/v1/chat/completions` 端点工作（HTTP 200）
- ✅ Python `openai` SDK 透过 `OPENAI_BASE_URL` 自动接通
- ✅ AI-Scientist-v2 `create_client('gpt-5.4')` 端到端测试通过（reply: "hello world"）

**代理可用模型**（来自 `/v1/models`）：
`gpt-5.2`、`gpt-5.3-codex`、`gpt-5.3-codex-spark`、`gpt-5.4`、`gpt-5.4-mini`、`gpt-5.5`。
所有这些已加入 `ai_scientist/llm.py` 的 `AVAILABLE_LLMS` 白名单。

## 在命令行调用项目的模型参数

由于代理仅有 gpt-5.x 系列，所有脚本的 `--model*` 参数都用 `gpt-5.4`（或想省钱用 `gpt-5.4-mini`）：

```bash
# Ideation
python ai_scientist/perform_ideation_temp_free.py \
  --workshop-file ai_scientist/ideas/handwritten_archive_ocr.md \
  --model gpt-5.4 \
  --max-num-generations 10 \
  --num-reflections 4

# Launch BFTS
python launch_scientist_bfts.py \
  --load_ideas ai_scientist/ideas/handwritten_archive_ocr.json \
  --idea_idx 0 \
  --writeup-type icbinb \
  --model_writeup gpt-5.4 \
  --model_citation gpt-5.4-mini \
  --model_review gpt-5.4 \
  --model_agg_plots gpt-5.4-mini
```

> 注：`bfts_config.yaml` 内 `agent.code.model` 默认是 `anthropic.claude-3-5-sonnet-...`（Bedrock）。如果不用 Bedrock，需要把它改成 `gpt-5.4`。这一步在"开始 BFTS 实验前"做。

## 推荐工作流（M5 Pro 24GB）

### ✅ 阶段 0：本机生成研究 ideas（无 GPU 需求）

```bash
python ai_scientist/perform_ideation_temp_free.py \
  --workshop-file ai_scientist/ideas/handwritten_archive_ocr.md \
  --model gpt-4o-2024-05-13 \
  --max-num-generations 15 \
  --num-reflections 5
```

输出 `ai_scientist/ideas/handwritten_archive_ocr.json`。**人工挑选 / 编辑** 1–3 个最值得做的 idea。

### ⚠️ 阶段 1：BFTS 实验（建议租云 GPU）

`launch_scientist_bfts.py` 默认假设 CUDA。在 Mac 上跑会遇到：

1. `torch.cuda.device_count() == 0` → worker 不会分配 GPU（manager 仍可启动，只是子代码全用 CPU/MPS）
2. LLM 写出的代码大概率包含 `.cuda()` / `device='cuda'` → **运行时报错**，BFTS 通过"调试节点"机制可能修复，但成功率不高
3. 24 GB 共享内存对 OS + browser + python 都要让出空间

**两个选项：**

#### A. 推荐：云 Linux GPU（最稳）
租 1×RTX 4090 / A10 / L4 (24GB) 即可，把整个 repo + ideas json 推上去：
```bash
python launch_scientist_bfts.py \
  --load_ideas ai_scientist/ideas/handwritten_archive_ocr.json \
  --idea_idx 0 \
  --writeup-type icbinb \
  --model_writeup o1-preview-2024-09-12 \
  --model_citation gpt-4o-2024-11-20 \
  --model_review gpt-4o-2024-11-20
```

#### B. 本机硬跑（实验性，预期失败率较高）
1. 在主题 md 中已经写了"必须用 MPS、禁止 .cuda()"等约束（idea 文件已包含）
2. 调小 `bfts_config.yaml`：
   - `agent.num_workers: 1`
   - `agent.stages.stage1_max_iters: 5`
   - `agent.search.num_drafts: 1`
   - `exec.timeout: 1800`
3. 启动前 export 一个环境变量提醒 LLM：
   ```bash
   export AI_SCIENTIST_DEVICE=mps
   ```
   （注意：项目代码不会自动读取此变量，仅作为你心智模型；真正的约束在 idea md 的 prompt 里）

## 关于 PDF

`面向手写档案的OCR智能识别方法研究_赵启延.pdf` 被 DRM 加密无法解析正文。已根据**标题**和**领域常识**（手写档案 OCR）撰写研究主题。如果需要更精准对齐论文细节，可以：
- 用知网客户端打开 → 复制目录/章节 → 粘给我
- 或导出文字版（去 DRM 后）再让我精修 `handwritten_archive_ocr.md`

## 常见数据集（MPS 友好）

- **CASIA-HWDB 1.0/1.1**（离线孤立汉字，~3.7M 样本，可子采样）
- **CASIA-HWDB 2.0–2.2**（手写文本行）
- **ICDAR 2013 Chinese Handwriting**
- **MTHv2**（中国古籍手写，需子集）
- HuggingFace `datasets` 上的小型手写中文集

可在 idea md 中明确指定，避免 LLM 选超大集合。

## 验证环境（已通过）

```python
import torch
torch.backends.mps.is_available()   # True
torch.randn(3,3, device='mps')      # OK
```
