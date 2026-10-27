# Orin NX 上的 GPU 编码侧车

验证日期：2026-10-27。本文记录**如何让 FieldMind 真正跑在 Orin NX 的 GPU 上**，以及为什么必须拆成两个 Python 进程。CPU 回退路径见 [orin_nx_deployment_validation.md](orin_nx_deployment_validation.md)。

> 结论：GPU 路径已打通并实测达标。单帧编码 **P50 1040 ms／P95 1060 ms**（CPU 回退为 5939 ms／6600 ms，快 **5.7×**），关键帧 ready **P50 1.54 s／P95 1.92 s**（目标 ≤3 s，**达标**）。GPU `float32` 的 fingerprint 与 CPU `float32` **完全相同**（`14c34b65…`），历史索引无需重建。

## 一、为什么必须拆进程

JetPack 5（L4T 35.x）是最新支持 CUDA 11.4 的 Orin 系统，而 NVIDIA 为该版本发布的 CUDA torch 轮子**只有 CPython 3.8**：

```
torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl   # CUDA 11.4
```

本仓库的 MCP 后端依赖 `mcp>=2.3`，它要求 **Python ≥3.10**。两者不能共用一个解释器：

| 组件 | Python | torch | 原因 |
|---|---|---|---|
| 后端／MCP | 3.11 | 2.9.1+cpu | `mcp` 需要 ≥3.10；CPU torch 够跑协议与索引 |
| 编码侧车 | 3.8 | 2.1.0a0+CUDA 11.4 | JetPack 5 唯一可用的 GPU torch |

因此：**3.8 进程持有 GPU 与模型，3.11 后端通过 loopback HTTP 请求向量**。

排查过程中确认的其它不可行路线（不必重试）：

- `download.pytorch.org` 的 cu118/cu126 aarch64 轮子：cu118 需 403／镜像只有 x86_64，cu126 需 CUDA 12 驱动，本机驱动只到 11.4。
- `pypi.jetson-ai-lab.io/jp5`：已 404，只剩 jp6/sbsa。
- 升级宿主机到 JetPack 6：会破坏现有传感器／导航栈，明确排除。
- CUDA 11.4 自建 cp311 torch：可行但要数小时编译且无官方支持，优先级低于侧车方案。

## 二、一次性环境准备

### 1. CUDA 11.4 运行时库

NVIDIA 的 L4T 镜像常只带驱动，不带 `libcudart`／`libcublas`／`libcudnn`。实测缺失时 `import torch` 报 `libcublas.so.* not found` 或 `libopenblas.so.0`。

```bash
bash scripts/setup_gpu_encoder_env.sh --install-cuda-runtime
```

该参数会添加 `repo.download.nvidia.com/jetson/common r35.6` 源并安装
`cuda-cudart-11-4`、`cuda-nvtx-11-4`、`libcublas-11-4`、`libcufft-11-4`、
`libcurand-11-4`、`libcusparse-11-4`、`libcusolver-11-4`、`libcudnn8`、
`libopenblas-dev`、`libomp-dev`（合计约 2.6 GB 安装后体积）。

### 2. Python 3.8 环境

```bash
bash scripts/setup_gpu_encoder_env.sh
```

脚本创建 `~/wdy/.venvs/fm-gpu`，安装 JetPack torch 轮子、`torchvision==0.16.2`
（最后一个 cp38 aarch64 轮子，`timm` 导入时需要）、`open_clip_torch==2.32.0`
（最后一个支持 py3.8 的版本）、`numpy<2`、`timm`、`einops`、`huggingface_hub<0.35`。

> 注意：`torchvision` 必须 `--no-deps` 安装，否则 pip 会用 PyPI 的 **CPU** torch
> 覆盖 CUDA 版。脚本末尾会重新校验 `torch.cuda.is_available()`。

验证：

```bash
LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64 \
  ~/wdy/.venvs/fm-gpu/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

实测输出：`2.1.0a0+41361538.nv23.06 Orin`。

## 三、启动与接入

### 1. 启动侧车

```bash
bash scripts/start_encoder_sidecar.sh
# encoder ready in 19.9s
# encoder sidecar on http://127.0.0.1:8791 (fingerprint 14c34b65d9c2c393, dim 1024, device cuda:0)
```

侧车默认读 `configs/jetson-gpu.yaml`（`device: cuda:0`、`dtype: float32`），
可用 `FIELDMIND_ENCODER_PORT`／`FIELDMIND_GPU_CONFIG` 覆盖。

### 2. 让后端使用它

```bash
.venv-jetson/bin/python -m slamibot_fieldmind.backend \
  --root /mnt/usb/slamibot_record_data/semantic_index \
  --model-config configs/jetson.yaml \
  --encoder-url http://127.0.0.1:8791 \
  --port 8765
```

MCP 服务同理：

```bash
.venv-jetson/bin/python -m slamibot_fieldmind.mcp_server \
  --root /mnt/usb/slamibot_record_data/semantic_index \
  --encoder-url http://127.0.0.1:8791 --preload-encoder
```

`/v1/health` 的 `environment` 会显示侧车真实设备，而不是后端所在主机：

```json
{"torch": "2.1.0a0+41361538.nv23.06", "cuda_available": true, "cuda_version": "11.4",
 "device": "cuda:0", "device_name": "Orin", "model_dtype": "torch.float32"}
```

### 3. 端到端冒烟

```bash
.venv-jetson/bin/python scripts/device_smoke.py \
  --root /mnt/usb/slamibot_record_data/semantic_index_gpu \
  --encoder-url http://127.0.0.1:8791 \
  --image /tmp/frame_1024x576.jpg --frames 20
```

## 四、实测数据

同一张 1024×576 帧，同一台 Orin NX（MAXN），同一 `input_resolution: [512]`：

| 指标 | CPU float32 | GPU float32 | GPU bfloat16 |
|---|---|---|---|
| 单帧编码 P50 | 5939 ms | **1040 ms** | 266 ms |
| 单帧编码 P95 | 6600 ms | **1060 ms** | — |
| 关键帧 ready P50 | 11.4 s | **1.54 s** | — |
| 关键帧 ready P95 | 18.7 s | **1.92 s** | — |
| 模型加载 | 12 s | 20 s | 20 s |
| 显存峰值 | — | 0.94 GiB | ~0.9 GiB |
| fingerprint | `14c34b65…` | **`14c34b65…`（相同）** | `b279c8f2…`（不同） |

bfloat16 快 4× 但 fingerprint 改变，会让既有索引失效，因此**默认用 float32**。

### 数值一致性

| 比较 | max abs diff | 最小余弦 |
|---|---|---|
| GPU float32 vs CPU float32（图像） | 2.98e-07 | 1.0000000 |
| GPU float32 vs CPU float32（文本） | 2.09e-07 | 1.0000000 |
| GPU bfloat16 vs float32 | 1.05e-03 | 0.999965 |

float32 的差异是单精度舍入级别，且 fingerprint 相同，因此**CPU 建的集合可由 GPU 侧车直接检索**（已实测：`semantic_cpu_fp32` 集合用侧车 `search` 返回 2 条候选）。

### 传输格式

`HttpEncoder` 传**原始 uint8 像素**（base64），不是 JPEG：

- JPEG q95 会引入 2.8e-4 的向量偏差（余弦 0.999999），虽然可忽略，但破坏了"与进程内完全一致"这一可验证性质。
- 原始像素在 loopback 上 ~3 MB/帧、耗时 ~1 ms，相对 1 s 的编码可以忽略。
- 实测传输后向量与进程内编码 **max abs diff = 0.0**。

## 五、注意事项

### 1. `torch.distributed` 缺失

NVIDIA 的 JetPack 轮子用 `USE_DISTRIBUTED=0` 编译，而 RADIO 在模块级 `import` 了
`torch.distributed.nn.functional` 与 `dist.is_initialized()`。侧车所在环境没有这些符号，
会导致 `import radio` 直接失败。

处理：`backbone._shim_torch_distributed()` 在导入 RADIO 前补齐 `is_initialized`／
`get_rank`／`ReduceOp`／`all_reduce` 等空实现。RADIO 只在多进程训练路径使用它们，
单进程推理不受影响。

### 2. `open_clip` 版本差异

`text_tower.py` 的 `encode_text` 原本调用 3.x 的 `text_global_pool(..., eos_token_id=...)`，
2.x 的签名没有该参数。现在用 `_pool_accepts_eos()` 检测签名，2.x 走 `argmax` 分支
（同为最后一个 token），3.x 保持原行为。两侧都已实测。

### 3. GPU 与 SLAM／导航共存

Orin NX 的 GPU 与 CPU 共享 16 GB 内存。侧车峰值约 0.94 GiB，加载期约 1.8 GiB，
未观察到对 SLAM 频率的影响（**但未做并机长时测试**，见下）。

建议：侧车 `nice -n -5` 或在其启动后再拉起 SLAM；两者不要同时冷启动。

### 4. 侧车不保管索引

侧车只做编码。索引、SQLite、图片仍由 3.11 后端独占写入，避免跨进程并发写。

## 六、仍未验证

1. **并机长时负载**：与 SLAM／导航共跑 30 分钟以上、再 2–4 小时，记录 SLAM 频率、温度、功耗。本次仅单独测编码吞吐。
2. **真实相机**：本机 `/SLB_CAM_A/compressed` 与 `/slam_pose` 无发布者，用静态帧验证。真实 OAK 采集下的端到端延迟未测。
3. **侧车崩溃恢复**：`HttpEncoder` 每次请求独立连接，侧车重启后后端会自动重连，但没有做故障注入测试。
4. **容器化**：`deploy/docker-compose.encoder.yml` 已写好，但本机 Docker 无 `nvidia` runtime，未实际运行。需先用 `nvidia-container-toolkit` 配置 runtime。

## 七、代码改动清单

| 文件 | 改动 |
|---|---|
| `src/slamibot_fieldmind/encoder_service.py`（新增） | py3.8 HTTP 侧车：`/health`、`/encode_images`、`/encode_texts` |
| `src/slamibot_fieldmind/remote_encoder.py`（新增） | py3.11 客户端 `HttpEncoder`，接口与 `AlignedEncoder` 对齐 |
| `src/slamibot_fieldmind/backbone.py` | `_shim_torch_distributed()` 补齐 `USE_DISTRIBUTED=0` 的缺口 |
| `src/slamibot_fieldmind/text_tower.py` | 兼容 `open_clip` 2.x/3.x 的 pooling 签名 |
| `src/slamibot_fieldmind/backend.py` | 新增 `--encoder-url`／`--encoder-timeout` |
| `src/slamibot_fieldmind/mcp_server.py` | 同上 |
| `scripts/device_smoke.py` | 透传 `--encoder-url` |
| `scripts/setup_gpu_encoder_env.sh`（新增） | py3.8 + CUDA torch 环境一键搭建 |
| `scripts/start_encoder_sidecar.sh`（新增） | 启动侧车（含 `LD_LIBRARY_PATH`／`PYTHONPATH`） |
| `configs/jetson-gpu.yaml`（新增） | 侧车专用：`cuda:0`、`float32` |
| `deploy/docker-compose.encoder.yml`（新增） | 侧车容器定义（未实测） |
| `tests/test_encoder_service.py`（新增） | 11 项：协议版本、原始像素往返、fingerprint 校验、BGR、错误载荷 |

回归：`python -m pytest -m "not slow"` → **249 passed, 13 skipped**。
