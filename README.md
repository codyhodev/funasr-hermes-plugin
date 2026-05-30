# FunASR STT Plugin — Hermes Agent

本插件为 Hermes Agent 提供本地语音识别（STT）能力，基于 **FunASR SenseVoiceSmall** 模型，GPU 加速，专注中文识别。

## 依赖

- **Hermes Agent**（已安装）
- Python 包：`funasr`、`torch`、`torchaudio`、`soundfile`
- 网络：首次使用需从 ModelScope 下载模型（~893MB）

## 安装（二选一）

### 方式一：目录插件（直接拷贝）

```bash
# 1. 复制插件到 Hermes 插件目录
cp -r src/funasr_plugin/ ~/.hermes/plugins/funasr/

# 2. 安装 Python 依赖
uv pip install funasr torch torchaudio soundfile

# 3. 启用插件
hermes plugins enable funasr
hermes config set stt.provider funasr
```

### 方式二：pip 本地安装（推荐）

```bash
# 1. 在项目根目录执行
pip install -e .

# 2. 启用
hermes config set stt.provider funasr
```

方式二通过 `hermes_agent.plugins` 入口点自动发现，无需手动拷贝到插件目录。

## 架构

```
┌──────────────┐      Unix Socket       ┌──────────────┐
│  Hermes CLI   │ ◄─────────────────────► │  funasrd     │
│  (TUI/Gateway)│                         │  (常驻进程)   │
│               │                         │  GPU 加载模型 │
│  引用计数      │                         │  处理转录请求  │
└──────────────┘                         └──────────────┘
```

- **funasrd.py** — 守护进程，预加载 SenseVoiceSmall 模型并监听 Unix Socket
- **provider.py** — Hermes TranscriptionProvider 实现，通过 Socket 通信
- **\_\_init\_\_.py** — 插件入口，自动拉启/关闭守护进程

## 特性

- ✅ **常驻进程** — 模型一次加载，多次使用（首次 ~7s，之后 ~150ms）
- ✅ **多组件共享** — Gateway + TUI 共用同一个守护进程，不重复占显存（~1GB）
- ✅ **引用计数** — 最后一个使用者退出时才真正关闭守护进程
- ✅ **自动清理** — 退出时删除 Socket 和 PID 文件

## 注意事项

- 插件默认使用 GPU（CUDA），无 GPU 时自动回退 CPU（速度较慢）
- 模型首次使用自动下载到 `~/.cache/modelscope/hub/`
- 如果之前用过 faster-whisper 的 .pth shim，请先删除旧文件再启用本插件
