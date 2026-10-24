# FieldMind 仓库与完整迁移

2026-10-24，视觉记忆后端正式命名为 **SLAMIBOT FieldMind**，寓意现场认知。源码、模型缓存、历史索引、关键帧图片及报告统一迁至新目录；原 `radio_retrieval` 项目目录已移入回收站，旧 Python 包已卸载，不保留兼容入口。

| 项目 | 当前名称 |
|---|---|
| Git 仓库／Python 分发包 | `slamibot-fieldmind` |
| Python 包 | `slamibot_fieldmind` |
| 后端命令 | `slamibot-fieldmind` |
| stdio MCP 代理命令 | `slamibot-fieldmind-mcp` |
| MCP 服务标识 | `slamibot-fieldmind` |
| Windows 工作目录 | `E:\slb\slb_code\OAK\slamibot-fieldmind` |
| Jetson 镜像工作目录 | `/opt/slamibot-fieldmind` |

## 数据与运行服务

`models/`、`data/`、`reports/` 与本地官方 RADIO 源码直接迁入新目录，不依赖指向旧目录的联接。Git 仅保存源码、配置、文档和模型来源清单；不上传权重、业务图片、索引、报告或凭据。

历史数据库结构、`radio_clip_v1` 集合标识、观测协议版本、编码器 `CODE_VERSION` 和指纹算法不因改名而变化，不重算历史 embedding。关键帧图片使用相对路径，迁移数据根目录后仍可读取。历史日志中的旧绝对路径作为当时运行记录保留，不应直接复制为启动命令。

本机原 `.venv-radio` 仅作为已安装 PyTorch 等依赖的 Python 环境继续使用，旧 `radio-retrieval` 分发包已卸载，重新可编辑安装新项目。该环境目录名指代 RADIO 推理依赖，不是旧项目的兼容入口；其他机器可自行选择虚拟环境名称。

运行入口统一为：

```bash
python -m pip install -e ".[backend,dev]"
python -m slamibot_fieldmind.backend --root data/backend --allow-unmounted
python -m slamibot_fieldmind.mcp_proxy --url http://127.0.0.1:8765/mcp
```

MCP 代理只连接后端，不加载模型。原独立相机模式如仍需使用，也通过 `slamibot_fieldmind.mcp_server` 调用。

## 设备框架

设备框架仓库继续维护 ROS 驱动、消息、任务控制与桥接，不迁入本仓库。接入分支的 Compose 命令、镜像挂载路径及文档统一引用 FieldMind：

```bash
export FIELDMIND_MODELS_DIR=/absolute/path/to/slamibot-fieldmind/models
export FIELDMIND_BACKEND_IMAGE=slamibot-fieldmind:jetson
```

HTTP 观测协议、MCP 地址和九个工具名保持不变。迁移运行服务时停止旧进程再启动新进程，避免重复占用相机／端口或同时写同一索引根目录。

## 本次验证

- 4 个 SQLite 数据库完整性通过，18 个 float16 向量分区及 1299 个迁移残留数据文件核验通过。
- 新目录下的后端恢复 901 帧 `ready` 索引，模型指纹仍为 `15e4b48686123c94d44e5dbf1585c5806c491bd7e7b5d13cb318ffc2881b69f1`。
- 236 项基础测试通过。首次将完整测试与模型冷启动同时运行时，两项 MCP 测试遇到 Windows 分页文件不足；停止并行冷启动后两项单独重跑通过。开发机应顺序执行冷启动和多进程测试。
- 经新 `slamibot_fieldmind.mcp_proxy` 验证九个工具及关键帧图片读取；原图和向量无需重建。
- 后端重启后当前视野缓存等待新的观测输入；历史索引可立即查询，不将旧缓存冒充当前视野。

本地核验记录为 `data/fieldmind_migration_20261024.json` 和 `reports/fieldmind_migration_mcp_20261024.json`，这些运行证据不随代码上传。
