# walk02 演示录制操作

更新日期：2026-10-22。演示使用虚拟机里的 `/home/wdy/bag/walk02.bag`，只发布 `/SLB_CAM_A/compressed`；Windows 运行现有 YOLO 网关和 RADIO／MCP 后端。

## 对话流程

| 用户指令 | Agent 操作 | 应展示的结果 |
|---|---|---|
| 开始任务 | 创建新会话、注册演示导航点；送入一张图像，等待实际 YOLO／后端接收，然后暂停 | 任务就绪、暂停；提示用户打开 RViz |
| 当前视野里有什么 | MCP `describe_view(camera_id="SLB_CAM_A")`，核对当前演示会话 | 单帧 YOLO 类别与数量；此步不需要 RADIO |
| 开始导航 | 恢复回放，持续更新缓存、关键帧与 RADIO 索引；报告经过的位置 | 图像、模拟轨迹、位置一至位置四 |
| 播放结束 | 等待转发和 RADIO 队列完成，确认 stopped／complete | “导航演示结束，有什么需要帮助的吗？” |
| 找一下哪里有人 | MCP RADIO 召回＋YOLO person 类别核验，取图 | 帧 ID、拍摄位姿、最近导航点、相对偏移和图片 |
| 哪里有四把椅子 | MCP 检索＋元数据补查，筛选同帧 4 个 chair 检测，再看图 | 检测数量、帧 ID、拍摄位姿和导航点关系 |
| 哪里有一个白色花瓶的盆栽 | MCP 语义召回；优先植物／花瓶框相互关联的候选，再查看图片核验白色 | 对应证据；区分类别检测与颜色视觉核验 |

收到“开始导航”后，Agent 保持当前回合运行直到回放、索引结束；可每经过一个点简短报告进度，不要启动后立即结束回合，否则不会自动反馈完成。遇到暂停指令则暂停时间线，恢复时不跳过等待期间的图像。

## 开场与导航点

默认从 CAM_A 开始后约 **21.7 秒**的办公区开场。原包前约 20 秒没有有效 YOLO 检测，不适合作为“当前视野有什么”的开场。这个偏移明确保存在 manifest 的 `clip_start_offset_s`，没有修改图像或检测结果。参数 `--start-offset 0` 可从包头播放。

准备阶段先送出一次开场图像并暂停，RViz 原图使用 latched 发布，晚启动也能收到。不会循环重发源图像，不会因等待用户而生成重复历史关键帧。默认剩余约 214 秒、1 倍速。

| 名称 | 开始导航后的媒体时间 | 演示地图坐标 |
|---|---:|---|
| 位置一 | 30 秒 | (5, 0, 0) |
| 位置二 | 90 秒 | (20, 0, 0) |
| 位置三 | 150 秒 | (35, 0, 0) |
| 位置四 | 210 秒 | (50, 0, 0) |

导航点绑定本次坐标会话，以唯一 `place_id` 登记到后端，MCP `list_places` 可读。RViz 用英文 Position 1～4 标注，避免默认字体缺少中文字形。距离计算只使用同一坐标会话，输出的是**拍摄相机相对导航点的位置**，不是目标三维位置。

本包 RTK 无定位，轨迹仍为模拟；图像、YOLO、RADIO 推理和索引是真的。此次没有执行真实导航动作，不把播放到某一时刻宣称为设备真实抵达。

## Windows 启动和控制

前置条件：虚拟机运行且 VMware Tools 可用，ROS 适配包已在 `/home/wdy/radio_replay_ws/devel_ros` 构建；Windows 后端 `192.168.1.4:8765` 模型 ready，CPU YOLO 网关 `192.168.1.4:8766` 可用。连接、依赖和构建见 [交接文档](agent_handoff_vm_ros_radio_yolo.md)。

以下命令在 `slamibot_fieldmind` 目录执行。`SLAMIBOT_VM_PASSWORD` 从已授权会话注入，脚本不保存口令。`start` 只启动到暂停阶段：

```powershell
..\.venv-radio\Scripts\python.exe scripts/demo_vm.py start
..\.venv-radio\Scripts\python.exe scripts/demo_vm.py status
..\.venv-radio\Scripts\python.exe scripts/demo_vm.py start-navigation
# 按需使用：
..\.venv-radio\Scripts\python.exe scripts/demo_vm.py pause
..\.venv-radio\Scripts\python.exe scripts/demo_vm.py resume
..\.venv-radio\Scripts\python.exe scripts/demo_vm.py stop
```

`start` 会同步控制脚本和 RViz 配置到 `/home/wdy/radio_demo`；生成独立输出目录 `/home/wdy/radio_replay_runs/demo_<run_id>`。Windows 当前 manifest 是 `data/demo_walk02/active_demo.json`，包含本次 session／coordinate ID、时间窗口、导航点和播放状态。`status` 会刷新它。

不要同时运行另一份占用 ROS 11322／控制 8767 的回放。控制端口只绑定客户机 `127.0.0.1`，宿主机经 VIX 在客户机内发请求。正常结束后进程保持查询／显示，`stop` 关闭本次 ROS 进程，后端继续运行。重录使用 `start` 新会话，不删除历史。

预演参数示例：`start --limit-seconds 8 --task-id walk02_demo_rehearsal`；全程加速预演可用 `start --speed 2 --task-id walk02_demo_full_rehearsal`。短于导航点时间的预演不登记那些点。正式录制不加限时／加速参数。

## Ubuntu 打开 RViz

收到“任务已就绪并暂停”后，在虚拟机桌面的终端执行：

```bash
source /opt/ros/noetic/setup.bash
source /home/wdy/radio_replay_ws/devel_ros/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11322
export ROS_IP=127.0.0.1
rviz -d /home/wdy/radio_demo/demo_walk02.rviz
```

配置中的图像显示使用 `/demo/camera/image_raw`，来源仅为 CAM_A 原始 JPEG 解码；轨迹是 `/demo/path`，导航点是 `/demo/navigation_points`，Fixed Frame 为 `simulated_map`。相机画面没有伪造检测叠框；检索证据的彩色叠框通过 MCP 图片接口返回。

## MCP 查询与证据

准备完毕后，以下工具脚本只调用现有 MCP，不加载模型。每次 `--out` 使用新目录：

```powershell
..\.venv-radio\Scripts\python.exe scripts/demo_mcp_query.py current `
  --out reports/demo_take01_current
..\.venv-radio\Scripts\python.exe scripts/demo_mcp_query.py person `
  --out reports/demo_take01_person
..\.venv-radio\Scripts\python.exe scripts/demo_mcp_query.py four-chairs `
  --out reports/demo_take01_chairs
..\.venv-radio\Scripts\python.exe scripts/demo_mcp_query.py white-vase-plant `
  --out reports/demo_take01_plant
```

脚本从 active manifest 限定本次会话、相机、历史时间范围，避免命中预演或其他任务。保存每次调用的参数和原始 JSON、候选选择及图片路径。相对导航点计算也取 MCP `list_places` 的结果，不直接查询 SQLite 代替 MCP。

“四把椅子”先查 RADIO 前 50 名，再用 `find_frames` 补查全会话的 chair 检测，防止满足数量的帧被语义 Top-K 排除。筛选是 Agent 侧步骤，原始 `matches` 排名保持原样；不能说 MCP 自动完成了数量排序。检测到 4 个框不保证实际场景恰好只有 4 把椅子，需查看图片，必要时说明可见的其他椅子或遮挡。

“白色花瓶的盆栽”优先查看植物与花瓶位置关联的图片；YOLO 没有颜色类别，脚本不会自动声称白色已验证。Agent 必须打开 MCP 图片确认颜色，不符合时继续查看候选或报告未找到可靠证据。

所有历史图像保留原始采集 header，MCP `stale=true` 是预期行为；当前模式指本次回放停住的视野，不是实时相机。回答时用“当前演示画面／历史观测”，保留真实时间和模拟位姿来源。RADIO 相似度与 YOLO 置信度分开说明。

## 状态与故障排查

状态依次为 `initializing → paused → playing → indexing → complete`；中途可 `pause／resume`，失败为 `failed`，主动结束为 `stopped`。`history_complete=true` 表示队列已完成；只收到控制命令响应不代表导航结束。

检查 Windows active manifest、客户机输出目录中的 `status.json`、`roscore.log`、`adapter.log`，以及同名目录旁的 `.launcher.log`／`.exit.json`。控制与查询不重复打开 OAK、不另起 RADIO 模型，不改变现有 YOLO 阈值。

代码入口：`scripts/demo_vm.py`（宿主机控制）、`demo_replay.py`（ROS 回放）、`demo_support.py`（时间线／位置／筛选）、`demo_mcp_query.py`（真实 MCP 证据链）、`config/demo_walk02.rviz`（显示配置）。

## 已完成的预演验证

已在虚拟机完成一次 2 倍速全片段预演：2142 张源图像、四个导航点均经过，保存 144 张关键帧且全部 ready／available；结束后会话 stopped、history_complete 为 true。起始暂停时只送出一帧，真实 MCP 返回 1 人、4 个 chair 检测框；人员、四个椅子检测框和白色花瓶盆栽均取得 MCP 图片，人员与白色花瓶经目视核对，椅子画面还存在其他未框出的椅子。

该数量是加速预演的实际结果，不是正式 1 倍速验收固定值。新增控制／证据筛选测试 9 项通过，全套非 slow 测试 236 项通过。RViz 配置已部署、依赖已确认；未替用户打开桌面 RViz，首次录制时需按上方命令确认显示。

验证摘要见 [walk02 Demo 预演记录](../reports/walk02_demo_rehearsal_20261022.json)。预演数据保留为独立会话，正式“开始任务”会新建会话并只查本次数据。
