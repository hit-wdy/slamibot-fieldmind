# ROS Noetic：walk02 回放与模拟位姿

回放素材统一使用 `/home/wdy/bag/walk02.bag`，图像只使用 `/SLB_CAM_A/compressed`。Ubuntu 虚拟机负责 ROS 回放、模拟里程计与转发；Windows CPU YOLO 网关负责离线检测，FieldMind 后端负责 RADIO、当前缓存、关键帧存储和 MCP。设备部署仍使用 OAK 板端 YOLO。

本机服务、模型和历史索引统一位于 `slamibot-fieldmind` 工作区。以下命令用于新建回放，不应重复启动已有后端；现有服务状态先读 `/v1/health`。完整迁移约定见 [迁移说明](repository_migration.md)，VMware 操作见 [Agent 交接](agent_handoff_vm_ros_radio_yolo.md)。

## 位姿与时间语义

该包已检查的 235 条 `/rtk/raw` GGA 均为质量 0、卫星数 0、经纬度为空，因此使用明确标记的模拟位姿。模拟节点根据每张图像的 header 时间发布 `/Odometry`，轨迹包含静止、平移和原地转向，用于验证关键帧选择。它不是 SLAM 结果，不能用于评价定位精度。

包记录时间、`/clock`、图像 header 时间分别保留，不混用为采集时间。回放专用 launch 使用合成的单位外参，body 和 camera 位姿均带 `simulated` 来源；设备生产环境仍要求真实外参。每次回放生成新的会话与坐标标识。

## Ubuntu 构建

将设备框架 `src/semantic_perception` 放进隔离工作区 `~/radio_replay_ws/src/`。不激活 Conda，执行：

```bash
source /opt/ros/noetic/setup.bash
cd ~/radio_replay_ws
catkin_make --build build_ros -j2 \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DEMPY_SCRIPT=/usr/lib/python3/dist-packages/em.py \
  -DPY_EM=/usr/lib/python3/dist-packages/em.py \
  -DCATKIN_DEVEL_PREFIX="$PWD/devel_ros"
source devel_ros/setup.bash
```

系统 Python 为 3.8；FieldMind 推理环境独立运行，不将其 Python 依赖装进 ROS 系统环境。

## Windows 后端与离线 YOLO

仅在未运行服务时，从新仓库根目录启动，`--root` 显式指定本次数据目录：

```powershell
..\.venv-radio\Scripts\python.exe -m slamibot_fieldmind.backend `
  --root data/walk02/index --allow-unmounted `
  --host 192.168.1.4 --port 8765
```

后端只监听指定地址，本机检查也使用 `http://192.168.1.4:8765/v1/health`。地址以现场可达的宿主机地址为准。确认 `model_state=ready` 后，另开终端启动离线 YOLO 网关：

```powershell
..\.venv-radio\Scripts\python.exe -m pip install -e ".[backend,replay-yolo]"
..\.venv-radio\Scripts\python.exe scripts/replay_yolo_gateway.py `
  --model models/yolov6/yolov6nr1_coco_512x288.xml `
  --upstream http://192.168.1.4:8765 --host 192.168.1.4 --port 8766
```

网关使用已有 YOLOv6 Nano R1 OpenVINO IR，对同一张 JPEG 推理后转发原图及检测，不持有 RADIO。模型摘要见 `models/assets.json`。当前解码使用 `objectness × class_score`，并应用分类门槛、NMS、重复框及冲突框过滤；参数以 `configs/detection_postprocess.json` 和网关 `--help` 为准。叠框使用分类颜色与大字号。

## Ubuntu 回放

将新版本 `scripts/replay_rosbag_validation.py` 复制到虚拟机，先 source ROS 和隔离工作区，再执行：

```bash
/usr/bin/python3 /tmp/replay_rosbag_validation.py \
  --bag /home/wdy/bag/walk02.bag --image-topic /SLB_CAM_A/compressed \
  --camera-id SLB_CAM_A --task-id walk02_bag_simulated \
  --pose-note 'RTK has no fix; using explicitly simulated pose' \
  --url http://192.168.1.4:8766 --port 11322 \
  --out /home/wdy/radio_replay_runs/walk02_new_run
```

输出目录必须是新目录。脚本启动隔离 ROS master、模拟位姿节点、图像适配节点及桥接，等待订阅后回放。结束后清空转发槽、停止历史接收、等待 RADIO 队列，并关闭本次启动的 ROS 进程。`report.json` 保存会话、消息计数、索引状态及异常，节点日志在相邻目录。

只播放 CAM_A，不播放 CAM_B／CAM_C、IMU、雷达或无效 RTK 原始串口消息。没有 YOLO 网关时可直接发送到 8765，但检测状态是 `unavailable`，不能称为“已检测且没有目标”。成功推理但没有目标是 `available` 加空列表。

需要暂停、恢复和导航点演示时，使用 [walk02 演示录制流程](walk02_demo_recording.md)，不要同时启动本文的连续回放。

## MCP 核验

```powershell
..\.venv-radio\Scripts\python.exe scripts/check_backend_mcp.py `
  --url http://192.168.1.4:8765/mcp --session-id <report.json里的session_id> `
  --camera-id SLB_CAM_A --out reports/walk02_mcp.json
```

MCP 始终访问后端 8765，不访问检测网关 8766。查询限定本次会话，避免多个回放混淆；图像时间来自历史包，当前缓存中的图片也应按其真实时间报告过期。查询先用 `list_sessions` 确定时间范围，不把回放时间当成实际拍摄时间。
