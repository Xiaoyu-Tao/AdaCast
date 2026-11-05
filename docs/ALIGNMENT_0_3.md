# AdaCast Lite 0.3：本轮对齐范围与结果

## 本轮范围

根据本轮要求，继续对齐预测与模型选择、Agent 业务流程、上下文管理、LLM 参数与工具调用语义、模型依赖和数据输入范围。

暂缓：任务 Skills 与跨任务记忆的完整生命周期、Critic 后处理（轨迹图、固定基线、历史模式检索等）、各厂商原生 Provider 接口。既有实现继续保留。本轮仍只通过 OpenAI-compatible Chat Completions 调用 LLM。

## 已实现的对齐

### Agent 业务流程

- `generate_model` 同时接受原版 `spec/hyperparams` 和 Lite `model/params`。实验不能通过顶层字段或嵌套参数改变冻结任务。
- 增加 `forecast_task read`、`evaluate_experiment` 和 `list_models`，分别读取冻结任务、验证绑定后的指标及可用模型依赖。
- `forecast_reflect` 支持原版 `effective/reason/next_direction/decision`，兼容 Lite 的 `rationale`。
- 成功和失败的实验都要记录反思；失败实验可以显式重试，每次保留独立 attempt_index。
- 实验预算按唯一配置计算，重试不重复占用配置名额；崩溃、超时和 IPC 错误分别记账，停止计数同步到持久状态。
- 历史投影保留真实尝试耗时、固定时间戳、失败原因和反思记录。上下文中的概要可通过工具展开为完整记录。
- 主角色遵守原职责和阶段门禁；命令调度器仍负责阶段推进，配置替代人工决策。

### 上下文

- 采用原版的字符 token 估计器；读取实际 usage，包含缓存 token 信息，并对序列化输入做容量检查。
- 对齐输入预留规则：存在独立 input limit 时预留默认 `min(20000, output limit)`；否则按 context 减 output limit 判断。
- 对齐工具结果裁剪的 40,000 token 保护区和 20,000 token 最小清理量，保护最近两个用户轮次及 Skill 结果。
- 工具输出截断改为 2,000 行 / 50 KiB；完整输出仍保存在审计中并可按偏移读取。
- 使用原 compaction 角色提示词和继续任务摘要模板；摘要使用 assistant 消息，保留最新完整用户轮次，并支持多批语义压缩。
- 压缩检查点校验原始历史哈希；原始输入输出不被删除，续跑复用已完成摘要。
- Provider 报告上下文溢出后，执行一次强制压缩并重放待处理轮次；禁止无限重复同一请求。
- 独立压缩模型可以声明自己的上下文/输出上限；压缩请求按该模型容量分批。

原版框架内部的消息部件、数据库和事件未迁移。Lite 保留自己的消息投影、SQLite、审计与恢复接口；模型输入的可用信息、裁剪规则和失败恢复是本轮对齐目标，不宣称每个请求的序列化字节完全相同。

### LLM 参数与协议

- 复用原模型名称对应的 temperature/top-p/top-k 默认规则，保留一份由参考实现生成的独立回归夹具。
- 支持全局及角色级采样参数、seed、reasoning_effort 和输出上限；角色设置优先。
- 可声明每个模型的 context、input、output 上限和 temperature/tool/top-k 能力；能力限制优先于请求。
- 默认输出上限改为原版的 32,000，并受模型声明上限约束；默认 Provider 自动重试改为 0，用户仍可显式配置。
- 对齐可修复工具名的大小写处理；保留可见 `reasoning_content`，并传入隔离窗口的响应部件。
- 区分上下文超限、输出截断、格式错误和可重试 HTTP 错误；输出截断不重复生成同一响应。
- 隔离窗口遇到全局预算或中断时，取消所属执行范围并传回原退出原因，避免当作普通窗口错误继续重试。

模型能力来自冻结配置，不依赖旧 Provider 数据库。原生厂商协议、OAuth 和插件钩子按要求暂缓。

### 模型与依赖

- 增加独立 `models` 命令和 Agent `list_models` 工具。列出实现、模型族、缺失依赖、安装 extra、基础模型仓库及是否需要权重下载。
- 模型名称统一规范化；TypeScript 与 Python 注册表有一致性测试。Lite 自有基线和 UserPrior 不再归入深度学习族。
- 提供 neural、foundation（Chronos）、timesfm、moirai、tirex、transformers-models、mamba、reformer 安装分组，所有分组均进入独立 uv 锁文件。
- 修复 Crossformer 对旧顶层 `models` 命名空间的引用。
- 源码环境和发布安装器都支持按需安装；科学计算覆盖约束随发布产物交付，Moirai 使用与参考项目相同的 uv override 原则。

TimesFM 安装分组选定 3.0.2，该版本声明和本项目需要的 `TimesFM_2p5_200M_torch` 接口可见其[官方项目声明](https://raw.githubusercontent.com/google-research/timesfm/master/pyproject.toml)和[官方导出接口](https://raw.githubusercontent.com/google-research/timesfm/master/src/timesfm/__init__.py)。这不等于本轮执行了权重下载或 GPU 预测。

`dependencies_available` 仅说明依赖存在并满足声明版本要求；硬件、权重、编译扩展及每个模型的全部参数组合仍需实际执行验证。

### 数据输入

- 支持 Pandas 日历频率，包括月初/月末、季度、年和工作日，以及带倍数的分钟/小时频率。
- `m/M` 按参考模型语义表示月，分钟使用 `min/t`，不再把 `m` 解析为分钟。
- 不规则时间序列沿用按观测行划窗的规则，不自动补点或重采样；`task.strict_frequency=true` 可启用严格连续性检查。
- 保留时间戳唯一、递增约束和固定切分；时间特征使用实际观测时间。
- 日历嵌入宽度依据频率特征确定；补齐年/月初/季度初等别名，已有小时级数值回归保持一致。

## 验证边界

独立参考夹具继续覆盖 ARIMA、Holt、DLinear、PatchTST、外部先验网格和反思校准。新增 HTTP、溢出恢复、裁剪、失败重试与原生 spec 接口测试；新增月度 Transformer 实际训练、日历/不规则窗口及依赖目录测试。

真实 LLM 的实验选择质量、所有模型的全参数组合、Foundation 权重下载和 GPU 推理仍未做完整对照。被明确暂缓的三类能力不计入本轮完成范围。

0.3 改变了冻结配置和恢复契约，旧 0.2 工作区应使用对应版本完成；不提供旧状态迁移。项目仍可独立安装、运行、恢复和交付。


## 本轮最终验证

- TypeScript：29 项通过，0 失败。
- Python（含 torch）：19 项通过，0 失败；已有原版预测、先验网格和校准夹具继续通过。
- 独立核心 Python 环境：16 项通过，3 项可选神经测试跳过，0 失败。
- 类型检查、schema、依赖独立性、uv 冻结锁和发布构建通过。
- 仅安装发布产物后，模型目录、运行、恢复、导出、交付模型推理、模拟 Agent 窗口协议通过。

机器可读记录：[scoped-0.3/verification.json](parity-results/scoped-0.3/verification.json)。实际 LLM 质量与 GPU 权重路径不包含在上述通过结论中。
