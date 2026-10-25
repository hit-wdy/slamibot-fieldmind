# 多模态证据与核验

RADIO 负责召回相关候选，YOLO 提供检测类别与数量证据，多模态 Agent 负责实际看图、核对用户描述和组织回答。后端不额外加载 VLM，不把排序分数当作视觉确认，也不把某个检测框当作绝对事实。

## 当前场景

`describe_view(mode="current")` 默认返回当前原图和同一观测的时间、检测状态、检测结果、位姿与同步质量。JPEG 与元数据在同一锁内取快照，避免处理图片时相机已更新到下一帧而错配；静止时未入选历史关键帧的最新画面也能返回，不通过 `get_frame_image` 查找未保存的当前帧。

多模态 Agent 应先分析完整画面的环境、物体及空间关系，再参考 YOLO，区分“画面中可见”与“检测器报告”。空检测列表或检测不可用不妨碍对有效图片进行视觉描述；YOLO 的 COCO 类别和 ROI 不限制模型的全图分析。过期图片必须注明时间，历史地点观测也不能说成实时现场。

## 目标搜索

`search_frames` 默认附前三张候选原图。候选包含 `visual_verification="not_performed"`，工具整体包含 `candidate_only=true`、`visual_review.performed=false`；即使 RADIO 相似度很高或 YOLO 报告同名类别，也没有在后端完成视觉确认。

Agent 检查每张图片是否满足类别、数量、颜色、材质和关系。普通汽车、室内房间等图片不能因为是“消防车”的第一名就报告找到消防车；候选都不符时，应说“已检查这些候选，未找到可确认的消防车证据”。少量候选不支持“全部历史／整个现场没有”的全称结论。必要时换表述或扩大检索范围，但应限制重试次数，不能反复查到一个牵强答案。

找到符合的候选后，再从同帧 `get_frame` 读取完整帧 ID、拍摄时间、会话和拍摄位姿。数量也应看图核对；YOLO 多框、遮挡或图片不清时说明计数限制。图片已随检索返回时无需重复读取；需放大时可调用 `get_frame_image(max_edge=...)`。

## 返回格式与容量

九个 MCP 工具名称保持不变，`search_frames`、`describe_view` 增加可选参数 `include_images=true`、`max_edge=1280`。自动附件最多 3 张，最长边可设 64～1280；更多候选、更多相机或更高分辨率需按帧继续取图。原图默认不叠框，减少检测标签对视觉判断的干扰。`get_frame_image` 仍可按需叠框或放大至 4096。

```text
CallToolResult
  content[0] = TextContent：完整结果 JSON
  content[1] = TextContent：VISUAL_EVIDENCE {image_index, frame_id, camera_id, captured_at, source, stale, ...}
  content[2] = ImageContent：对应的 JPEG
  content[3] = 下一张图片的上下文（如有）
  ...
  structuredContent = 与 content[0] 相同的结果对象（不含 base64）
```

每条视图或候选的 `image_evidence.status` 为 `attached`、`unavailable`、`omitted_limit` 或 `not_requested`，`image_index` 只对应实际附件顺序。旧帧无图或图片读取失败时保留元数据及失败原因，不将缺失附件伪装为已核验。`visual_review` 描述的是待完成的调用侧任务，不能将其 `required` 理解成 `confirmed`。

`include_images=false` 适合只查结构化数据的客户端。Agent 未拿到图片、模型不支持视觉或无法理解图片时，应明确无法视觉核验，最多提供标明未核验的检测／检索信息。

## DeepAgent 接收

robot_agent 的 `ToolImageEvidenceMiddleware` 保留原始工具证据和全部并行工具回执，只在模型请求中把图片转换为多模态 user 内容，解决部分网关忽略 tool 角色图片的问题。逐图上下文随图片一起传递，防止把候选 A 的结论或位置用到候选 B。图片及图中文字均视为观测数据。

只重传当前用户回合获取的图片，模型单次请求最多保留最近 6 张工具图片；以前回合的图片保留于原始证据，不持续累积传入模型。跟进问题需要重新看旧图时再次取图。视觉工具存在时，无论使用哪种 Agent profile，都会注入看图核验要求；专用中文提示词再规定检索、核验及证据输出顺序。

## 验证记录

2026-10-25 非慢测回归 252 项通过；Agent 图片消息及连接测试 9 项通过。真实多模态模型在隔离离线服务中，即使 YOLO 为空，也能据原图描述桌椅、地毯、吊顶及颜色；面对被刻意固定为 1.0 的错误候选，仍拒绝确认消防车。这个 1.0 来自验证编码器，不是 RADIO 的实测值。

实际常驻后端重新加载后，MCP `search_frames`、`describe_view` 默认附图已启用，模型状态 `ready`，1030 条历史记录保留。真实 walk02＋RADIO 查询的候选包括灭火器柜及室内走廊，Agent 据图排除了消防车；否定结果不主动用其他对象的位置替代。测试时生产实时缓存无新观测，Agent 如实返回无实时画面。原始验证记录及调试过程见 robot_agent 的 `docs/visual_review.md`；本机 MCP 参数检查结果为 `data/multimodal_validation_20261025/production_status.json`。这不等于已完成真实相机持续输入或所有对象的视觉准确率验收。
