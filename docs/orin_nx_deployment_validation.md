# Orin NX 16GB 实机部署验证记录

验证日期：2026-10-26。目标是按 [Orin NX 部署适配评估](orin_nx_deployment_assessment.md) 在本机完成部署验证。本文只记录**实测结果**与**修正**；评估文档中的建议若未被实测推翻，仍以评估为准。

> 重要结论：本机实际是 **JetPack 5.1.4 / L4T 35.6.4**，不是评估文档假设的 JetPack 6.2。由此导致 `pyproject.toml` 声明的 `torch>=2.8`、Python ≥3.10 无法在 GPU 上满足。本次部署验证以 **CPU 回退路径**跑通端到端流程，并把 GPU 阻塞点、内存优化和可复现步骤全部固化到代码与本文。CPU 回退可支撑功能验收，但**不满足实时索引性能目标**。

## 一、实测环境基线

| 项目 | 实测值 | 与评估文档的差异 |
|---|---|---|
| 设备 | NVIDIA Orin NX Developer Kit | 一致 |
| L4T | R35.6.4（`nvidia-l4t-core 35.6.4`） | 评估按 JetPack 6.2／L4T r36 假设 |
| JetPack | **5.1.4**（`nvidia-jetpack` 元包未安装，仅有 L4T 组件） | 评估假设 6.2 |
| OS | Ubuntu 20.04.6 LTS（Focal） | 评估假设 22.04 |
| 系统 Python | **3.8.10** | 评估要求 ≥3.10 |
| CPU／内存 | 8 核，15.5 GiB 可用，swap 7.6 GiB | 16GB 一致 |
| GPU 驱动 | `nvidia` 模块 35.6.4，CUDA driver API **11040（11.4）** | 关键：驱动只到 CUDA 11.4 |
| CUDA 运行时 | **未安装** `libcudart`／`libcublas`／`libcudnn` | 评估未验证 |
| 功耗模式 | `MAXN`（nvpmodel id 0） | 评估要求核实，已完成 |
| Docker | 28.1.1，runtime 仅 `runc`（**无 nvidia runtime**） | 评估要求核实 |
| 数据盘 | `/mnt/usb` 已挂载（466G，388G 可用），可写 | 一致 |
| 系统盘 | `/` 233G，验证后 84% 使用 | 需要注意 |
| ROS | Noetic + Foxy 桥运行中；OAK 话题 `:8766` **未启动**，`/SLB_CAM_A/compressed`、`/slam_pose` 无发布者 | 现场未接相机 |

基线探测命令（只读）已写入 `reports/platform_probe.json`：

```bash
python3 scripts/platform_probe.py --out reports/platform_probe.json
```

## 二、关键问题与解决

### 1. 镜像不是 JetPack 6.2，GPU 推理在当前宿主机不可用

**问题**：评估文档按 JetPack 6.2（L4T r36）设计，要求 torch ≥2.8。实测：

- 驱动 API 为 CUDA 11.4（`cuDriverGetVersion` 返回 11040）。
- NVIDIA 官方 JP5 redist 只有 `torch-2.1.0a0+...cp38`（Python 3.8／CUDA 11.4）。
- PyTorch 官方 aarch64 wheel 在 `cu126` 下要求驱动 12.x，与本机 11.4 不兼容；`cu118` aarch64 轮子在官方源被 403 拒绝，镜像源仅有 x86_64。
- `pypi.jetson-ai-lab` 的 `jp5` 索引已下线（404），只剩 `jp6`。

**处理**：

- 不升级宿主机（评估明确禁止，且会破坏现有传感器／导航栈）。
- 新增 `device: auto` / `text_encoder_device: auto`，在无 CUDA torch 时明确回退 CPU，并把**实际设备与 dtype**写进 `/v1/health` 的 `environment` 与 `model_warnings`，避免“看起来 ready、实际跑在 CPU”被误判（`src/slamibot_fieldmind/backbone.py`）。
- `deploy/Dockerfile.jetson` 去掉无意义的 `ARG BASE_IMAGE` 默认值假设，改为明确要求选 JetPack 6 基础镜像，并注明 JP5 无 torch≥2.8 构建。
- CPU 上 `bfloat16` 会走未向量化慢路径（实测 **70 s/帧**），因此加载器在 CPU 自动降级到 `float32` 并记录告警（实测 **5 s/帧**，快 14 倍）。

**遗留**：要满足实时性能目标，需二者之一：升级到 JetPack 6.x 并安装匹配的 CUDA torch，或用现有 nav 容器的 CUDA 11.4 环境单独构建 cp38 torch 后端。本次未擅自升级宿主机。

### 2. Hugging Face 直连不可达，模型无法下载

**问题**：设备上 `huggingface.co` 与 `download.pytorch.org` 均拒绝连接，只有 `hf-mirror.com` 可用；`scripts/fetch_assets.py` 把清单里的原始 URL 写死，下载必然失败。

**处理**：让脚本跟随 `HF_ENDPOINT`（与 `huggingface_hub` 同一开关），清单里的 origin 仍保留可审计，SHA-256 才是下载授权依据：

```bash
HF_ENDPOINT=https://hf-mirror.com python scripts/fetch_assets.py \
  --asset radio_v2.5_b --asset dfn5b_clip_vit_h14_378 --check
```

实测下载并校验通过：RADIO 649 MB、DFN5B CLIP 3.95 GB。

### 3. 冷启动峰值（评估第 2 条）已被实测确认，并已修复

**问题**：原实现先构建完整 CLIP（含 2.5 GB 视觉塔），再 `oc_model.visual = None`。

**实测（CPU float32，同一台设备）**：

| 加载方式 | 冷启动耗时 | 进程峰值 RSS | 文本向量 |
|---|---|---|---|
| `full`（原实现） | 21–24 s | **8.9 GB** | 基准 |
| `text_only`（新实现） | **12–13 s** | **4.1 GB** | **逐元素相同（max abs diff = 0.0）** |

实现：新增 `src/slamibot_fieldmind/text_tower.py`，只构建 CLIP 文本塔，并从固定 checkpoint 中只加载非 `visual.*` 键（294/686 个张量，1.42 GB）。配置新增 `model.text_encoder_load: text_only|full`，默认 `text_only`。用 `tests/test_text_tower.py` 证明文本向量与 fingerprint 和完整加载**完全一致**，向量空间不变，历史索引无需重建。

这直接回应评估“离线制作文字塔专用资源”的建议，且不需要新增离线资源文件——从同一固定 checkpoint 中裁剪即可。

### 4. 数据根目录被硬编码为 `/mnt/usb`

**问题**：`BackendRuntime` 无论 `--root` 指向哪里都先检查 `/mnt/usb`，NVMe 或容器内 `/data` 布局无法通过。

**处理**：`BackendConfig` 新增 `mount_path` 与 `data_root_is_mounted()`，校验**实际数据根是否位于声明的挂载卷**，而不是某个固定路径；新增 `--mount-path` 参数。原来的 `/mnt/usb` 仍为默认值，不传参数时行为不变。测试见 `tests/test_data_root.py`。

### 5. DepthAI 硬导入阻断后端与测试

**问题**：`capture.py` 顶层 `import depthai`，导致设备后端环境、MCP 代理和测试套件都必须安装 DepthAI。设备侧 OAK 由 SLAMIBOT 驱动持有（DepthAI 2.30），后端不应再装一份。

**处理**：改为按需导入 `depthai()`，仅在独立相机模式真正打开相机时要求该依赖。后端、MCP 代理与测试因此可脱离 DepthAI 运行（`src/slamibot_fieldmind/capture.py`）。

### 6. YOLO 归档缺失阻断纯几何测试

**问题**：`YoloV6Detector.__init__` 强制要求 VPU blob 存在，导致只验证坐标映射的测试失败，而这些测试不需要模型。

**处理**：显式传入的归档仍立即校验；默认归档推迟到 `attach()`（真正要上传到 VPU）时校验。错误信息和 `fetch_assets` 提示保留。

## 三、端到端验证结果

新增 `scripts/device_smoke.py`，在**本机**跑完整链路：启动后端 → 建会话 → 上传 JPEG 观测 → 关键帧落盘 → RADIO 索引 → HTTP／MCP 查询。

```bash
.venv-jetson/bin/python scripts/device_smoke.py \
  --root /mnt/usb/slamibot_record_data/semantic_index_smoke \
  --model-config configs/jetson.yaml \
  --port 8799 --frames 4 \
  --image /mnt/usb/slamibot_record_data/704bangongA/preview.jpg
```

实测（CPU 回退，`configs/jetson.yaml`）：

| 项目 | 结果 |
|---|---|
| 模型加载 | `ready`，约 12 s，峰值 4.1 GB |
| 观测接收 | 4/4 `accepted`，0 duplicate／0 failed |
| 索引 | `states={'ready': 4}` |
| 单帧编码 | P50 **4835 ms**，P95 **5561 ms** |
| 关键帧 ready 延迟 | P50 **10.6 s**，P95 **15.3 s** |
| MCP 工具 | 9/9 可调用（`search_frames`…`list_places`） |
| 语义检索 | 返回 2 条候选，附带 2 张原图 |
| 图片证据 | `get_frame_image` 返回 38,649 B JPEG |
| stdio 代理 | `slamibot-fieldmind-mcp` 转发 9 个工具，正常 |
| 重启恢复 | 杀进程重启后仍为 `states={'ready': 4}` |
| 磁盘布局 | `records.sqlite`＋WAL、`images/kf-xx/*.jpg`、`vectors/`，与文档一致 |

对照评估中的目标：**P95 当前缓存年龄 ≤1 s**（未测，需真实相机流）、**关键帧 ready ≤3 s**（CPU 为 10–15 s，**未达标**）、**SLAM 频率下降 ≤10%**（未测，需并机）。CPU 回退明确不满足实时目标。

### 稳定性与内存

2 分钟“图像编码＋并发文本请求”实测（CPU）：

- 稳定 RSS 约 2.5 GB，峰值 4.07 GB，结束后维持。
- 整机 `MemAvailable` 从 10.4 GB 降到约 10.39 GB（基本不下降）。
- 进程加载后编码期间未观察到内存持续增长。

## 四、当前建议的部署方式（在本机）

在 GPU 推理不可用的前提下，功能验证与现场试运行使用宿主 CPU 路径：

```bash
cd /home/jetson/wdy/slamibot-fieldmind

# 1) 依赖（Python 3.11 + CPU torch，已在本机验证）
uv venv --python 3.11 .venv-jetson
uv pip install --python .venv-jetson/bin/python torch==2.9.1 torchvision==0.24.1
uv pip install --python .venv-jetson/bin/python -e '.[backend,dev]'

# 2) 固定源码与模型（镜像加速）
git clone https://github.com/NVlabs/RADIO.git third_party/RADIO   # 或 codeload，见下
git -C third_party/RADIO checkout 4ffaf15d0ff48197c194e306772bdb5fb92db230
HF_ENDPOINT=https://hf-mirror.com .venv-jetson/bin/python scripts/fetch_assets.py \
  --asset radio_v2.5_b --asset dfn5b_clip_vit_h14_378

# 3) 启动（jetson.yaml 已设为 auto 设备）
.venv-jetson/bin/python -m slamibot_fieldmind.backend \
  --root /mnt/usb/slamibot_record_data/semantic_index \
  --model-config configs/jetson.yaml --port 8765
```

说明：

- GitHub 直连 codeload 在本机可用（约 1 MB/s）；`github.com` 的 git 协议会超时，可用 `https://codeload.github.com/NVlabs/RADIO/tar.gz/<commit>`。
- 数据目录必须是挂载卷内的路径；容器内用 `--mount-path /data`（`deploy/docker-compose.semantic.yml` 已带该参数）。
- 不传 `--mount-path` 时默认仍校验 `/mnt/usb`，与本机现状一致。
- CPU 上线程数影响有限（8 线程与默认差异 <6%）；分辨率影响显著：`[512]` 5.3 s、`[384]` 2.7 s、`[256]` 1.2 s，但降低分辨率会改变向量空间（fingerprint 改变），必须用新 collection 重建索引。

## 五、仍需在真实硬件上完成的事项

1. **GPU 路径**：升级到 JetPack 6.x 或用 CUDA 11.4 环境重编 cp38 torch 后，用 `configs/jetson.yaml` 的 `device: cuda:0` 复核真实吞吐与峰值。当前无 profiler 数据，不能引用 157 TOPS 推算。
2. **单路实采**：本机 `/SLB_CAM_A/compressed` 无发布者，未接 OAK／SLAM；需接入真实 CAM_A、`/slam_pose`、RTK 后复测时间关联、外参与会话。
3. **并机负载**：与 SLAM／导航共跑 30 分钟起、再 2–4 小时，记录 SLAM 频率与温度、功耗。
4. **导航 MCP／语音分流**：评估第 4 条的 `speak` 工具、`success:false` 语义、TTS 单次播报归属仍属 robot_agent／导航侧改造，不在本仓库验证范围。
5. **许可**：RADIO 与 DFN5B 的研究用途限制未变，商用发布前必须单独解决。
6. **容器**：本机 Docker 无 `nvidia` runtime，`deploy/Dockerfile.jetson` 未在本机构建；JP6 基础镜像拉取与构建需在有 r36 的设备或升级后进行。

## 六、本次代码改动清单

| 文件 | 改动 |
|---|---|
| `src/slamibot_fieldmind/text_tower.py`（新增） | 只加载配对的 CLIP 文本塔；复用官方分词器与权重，向量空间不变 |
| `src/slamibot_fieldmind/config.py` | 新增 `text_encoder_load`，校验取值 |
| `src/slamibot_fieldmind/backbone.py` | `auto` 设备解析、CPU 半精度回退、接入 text-only 加载、记录实际 dtype／告警 |
| `src/slamibot_fieldmind/backend_runtime.py` | 挂载校验跟随真实数据根；health 暴露 `environment`／`model_warnings` |
| `src/slamibot_fieldmind/backend.py` | 新增 `--mount-path` |
| `src/slamibot_fieldmind/capture.py` | DepthAI 按需导入 |
| `src/slamibot_fieldmind/detect.py` | 默认 YOLO 归档推迟到 `attach()` 校验 |
| `scripts/fetch_assets.py` | 支持 `HF_ENDPOINT` 镜像 |
| `scripts/device_smoke.py`（新增） | 设备端 HTTP＋MCP 端到端冒烟与报告 |
| `configs/jetson.yaml` | `auto` 设备、`text_only` 文本塔、CPU 安全的 dtype |
| `deploy/Dockerfile.jetson` | 明确 JetPack 6 基础镜像要求、线程与挂载参数 |
| `deploy/docker-compose.semantic.yml`（新增） | host 网络、数据卷、日志轮转、健康检查 |
| `tests/test_text_tower.py`、`tests/test_data_root.py`（新增） | 文本塔等价性与数据根校验回归 |

回归结果：`python -m pytest -m "not slow"` 全绿；`tests/test_text_tower.py`（需真实权重）全绿。
