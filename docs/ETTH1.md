# 运行 ETTh1

以下命令默认从包含 `dataset/` 和 `adacast-lite/` 的目录执行；Agent 小节会进入项目目录。输入文件为 `dataset/ETTh1.csv`；示例中的相对数据路径按配置文件所在目录解析。独立安装后也可把 CSV 放到任意位置并修改 `task.dataset_path`。

## 固定 DLinear 实验

```bash
uv sync --project adacast-lite/python --frozen --extra neural
adacast-lite/bin/adacast-lite validate --config adacast-lite/examples/task.etth1.local.json
adacast-lite/bin/adacast-lite run --config adacast-lite/examples/task.etth1.local.json --workspace adacast-lite/workspaces/etth1-dlinear
```

该模式不调用 LLM。默认使用 seed=42、5 个 epoch、batch_size=32、learning_rate=0.0001、moving_avg=25。当前固定模式使用模型默认超参数，不提供命令行 `--epochs` 等选项；Agent 模式可通过实验工具指定超参数。

## Agent 流程

先安装上述 neural 依赖。Agent 配置使用 OpenAI-compatible 接口 `https://api.kabuai.cn/v1`、模型 `deepseek-v4.1-flash` 和 `temperature: 0.7`，通过 `api_key_env: "LLM-API-KEY"` 引用 `adacast-lite/.env` 中的现有密钥。以下命令在子 shell 中进入项目目录，使 Bun 自动加载 `.env`：

```bash
(
  cd adacast-lite
  bin/adacast-lite validate --config examples/task.etth1.agent.json
  bin/adacast-lite run --config examples/task.etth1.agent.json --workspace workspaces/etth1-agent --json
)
```

该模式运行 Planner、Forecaster 和 Critic，在 DLinear/Holt/ARIMA 中搜索，允许 Agent 决定调整策略，并执行窗口反思。它需要真实 LLM，会产生 API 调用；最终输出可能包含调整和窗口推理结果。

## 配置与指标口径

两份配置均使用目标 `OT`、小时频率、seq_len=96、label_len=48、pred_len=96、gap=0、stride=96、0.6/0.2/0.2 切分。所有完整数值列参与模型输入；预测和评分只针对 `OT`。验证部分前半留给调整证据，后半用于模型验证。

该任务切分与本项目先前对齐实验相同；固定 DLinear 示例使用默认训练超参数，不等同于此前 2 epoch、lr=0.001 的显式参数对照。也未切换为其他 ETTh1 benchmark 切分口径。指标在训练拟合的标准化目标尺度上计算；预测 CSV 同时保存原始单位数值。

## 恢复、查看与导出

```bash
adacast-lite/bin/adacast-lite status --workspace adacast-lite/workspaces/etth1-dlinear --json
adacast-lite/bin/adacast-lite resume --workspace adacast-lite/workspaces/etth1-dlinear
adacast-lite/bin/adacast-lite export --workspace adacast-lite/workspaces/etth1-dlinear --output deliveries/etth1-dlinear
```

结果位于工作区 `.forecast/best-model/`：`report.md` 为指标和分析，`predictions/raw.csv` 和 `predictions/final.csv` 为预测，`manifest.json` 为指标、来源与文件哈希。每次新任务使用新工作区；已有任务使用 `resume`。
