import path from "node:path"
import { Store, lockWorkspace } from "../src/storage"
import { ForecastService } from "../src/forecast"
import { buildDelivery, verifyBundle } from "../src/delivery"
import { canonical, readJSON } from "../src/util"

// Run only after summarize-yaguo.py independently verifies the completed run.
const root = path.resolve(import.meta.dir, "..")
const workspace = path.join(root, "workspaces/yaguo-userprior-20261104")
const summary = await readJSON(path.join(root, "experiments/yaguo-userprior/summary.json"))
if (!Object.values(summary.checks).every(value => value === true)) throw new Error("Independent verification is incomplete")
const change = (gain: number) => `${gain >= 0 ? "下降" : "上升"} ${(Math.abs(gain) * 100).toFixed(2)}%`
const unlock = await lockWorkspace(workspace)
let store: Store | undefined
try {
  store = new Store(path.join(workspace, ".forecast"))
  if (store.get<any>("state")?.lifecycle !== "succeeded") throw new Error("Complete the workflow before reconciling its report")
  const old = await verifyBundle(path.join(store.root, "best-model"))
  if (canonical(old.final_metrics) !== canonical(summary.full_test_standardized_scale.adacast_final)) throw new Error("Verified summary differs from completed delivery")
  const original = summary.full_test_original_scale
  const rows = [["原始提交先验", "prior"], ["锁定规则调整", "adjusted"], ["AdaCast 最终输出", "adacast_final"]]
    .filter(([, key]) => original[key])
    .map(([label, key]) => `| ${label} | ${original[key].mse.toFixed(6)} | ${original[key].mae.toFixed(6)} | ${(original[key].wape * 100).toFixed(4)}% |`).join("\n")
  const candidate = summary.validation.candidate_evaluations.find((item: any) => item.candidate_id === "huabei-late-night-v1")
  if (!candidate?.acceptance_decision?.accepted) throw new Error("Locked candidate lacks passing validation evidence")
  const counts = summary.daily_mse_comparison
  const concentration = summary.gain_concentration
  const other = concentration.excluding_largest_gain_day
  const regression = summary.regression_concentration
  const sensitivity = summary.mse_improvement_ratio < 0 && regression
    ? `事后敏感性分析：MSE 增量最大的一天为 ${regression.largest_loss_day}，该日先验 MSE 为 ${regression.day_prior_mse.toFixed(6)}，最终 MSE 为 ${regression.day_final_mse.toFixed(6)}，占全部净 MSE 增量的 ${(100 * regression.largest_day_share_of_net_increase).toFixed(2)}%。同时从先验和最终结果中去掉该日，余下 278 天的先验 MSE 为 ${regression.excluding_largest_loss_day.prior_mse.toFixed(6)}，最终 MSE 为 ${regression.excluding_largest_loss_day.final_mse.toFixed(6)}，相对变化为${change(regression.excluding_largest_loss_day.mse_improvement_ratio)}。`
    : `事后敏感性分析：MSE 差值最有利的一天为 ${concentration.largest_gain_day}。同时从先验和最终结果中去掉该日，余下 278 天的先验 MSE 为 ${other.prior_mse.toFixed(6)}，最终 MSE 为 ${other.final_mse.toFixed(6)}，相对变化为${change(other.mse_improvement_ratio)}。`
  const reported = summary.llm_usage_by_status.reduce((sum: number, row: any) => sum + row.provider_reported_tokens, 0)
  const reserved = summary.llm_usage_by_status.reduce((sum: number, row: any) => sum + row.reserved_tokens_for_unknown_usage, 0)
  const content = `本次按 dataset/prompt_yaguo.md 完成华北负荷 OT 实验。唯一模型为 UserPrior；真实数据与先验均使用实验设置中的原始绝对路径，未使用项目中的 _21exog.csv 替代文件，未搜索、训练或生成其他模型先验。先验 SHA-256 为 ${summary.provenance.prior_sha256}。

训练截至 2025-06-30 09:30，共 157,543 点；第二个切分点为 2025-09-30 09:30。每个窗口为 711 点负荷历史、57 点 gap、96 点预测，label_len=711、stride=96。目标网格显式锚定 2025-07-01 00:00，因为提交先验在 gap 内也有替换值，不能用首个变化值推断目标起点。验证前半段为 7 月 1 日至 8 月 14 日，共 45 个拟合任务；独立后半段为 8 月 16 日至 9 月 29 日，共 45 个验收任务。8 月 15 日和 9 月 30 日的完整标签窗口跨越相应边界，按冻结的完整窗口规则排除。上方自动表的 Validation 指后半段 45 个任务。

测试保留 2025-10-01 至 2026-07-06 全部 279 天、26,784 点。第一个任务的真实历史为 9 月 23 日 00:00 至 9 月 30 日 09:30，预测 10 月 1 日；最后一个任务真实历史截至 2026 年 7 月 5 日 09:30，预测 7 月 6 日。gap 内只有提交先验及已知外生变量，未读取 gap 或预测日的真实 OT。

已自行撰写并冻结 knowledge/yaguo-domain.md。城市使用训练段 OT 与各城市 t2m/q/ssrd 的绝对 Pearson 相关均值预选，选出秦皇岛市，分数 0.116458。该分数仅是弱相关代理，不代表因果权重或测试最优。暴露秦皇岛市三项气象列及 18 个全局天气/事件字段，共 21 列；历史外生只取最近 288 点，即连续 72 小时，未来外生覆盖 gap 57 点和目标 96 点。负荷历史保持完整 711 点。全部 ${summary.exogenous_scope.checked_blind_contexts} 个盲上下文已核对范围。

验证前半段生成并比较三个轻量规则候选，最后锁定 huabei-late-night-v1 / v001，仅在 23:00—23:45 的四个点各增加 210 源单位。独立验证的原始 MSE 为 ${candidate.raw_metrics.mse}、MAE 为 ${candidate.raw_metrics.mae}；调整后 MSE 为 ${candidate.adjusted_metrics.mse}、MAE 为 ${candidate.adjusted_metrics.mae}，分别上升 ${((candidate.adjusted_metrics.mse / candidate.raw_metrics.mse - 1) * 100).toFixed(4)}% 和 ${((candidate.adjusted_metrics.mae / candidate.raw_metrics.mae - 1) * 100).toFixed(4)}%。现有验收允许 MAE/WAPE 最多 0.5% 回退，该候选通过的是容差门槛，没有证明验证改善。其余两个候选未选中。原始与调整后两组验证反思均为 enabled_no_trigger，没有触发逐窗修正；测试规则、知识和政策在查看测试误差前锁定。

测试集原始负荷尺度指标如下：

| 阶段 | MSE | MAE | WAPE |
| --- | ---: | ---: | ---: |
${rows}

反思阶段的不可变初始数值状态来自原始 UserPrior，锁定规则作用后的预测作为经验参考。表中分别列出规则独立应用结果和最终反思结果；控制器记录里的 baseline_kind=skill_adjusted 是策略标签，其数值基准指标与原始先验一致。全部 279 个初始状态均按原始负荷单位逐点核对原始先验。

完整测试最终 MSE 相对原始先验${change(summary.mse_improvement_ratio)}，MAE ${change(summary.mae_improvement_ratio)}。逐日 MSE 为 ${counts.improved} 天改善、${counts.worse} 天变差、${counts.unchanged} 天不变。每天所有预测与先验差异均不超过 0.01 源单位时计为不变，以避免 float32 还原噪声；完整测试聚合指标保留全部精度。原始尺度 WAPE 是 sum(abs(error))/sum(abs(actual))；上方自动表中的 WAPE 则作用于训练标准化目标，两者口径不同。KPIE 未提供业务公式，本次未虚构指标定义。

${sensitivity}该分析没有改变主要测试口径，全部 279 天仍保留在主指标内。

全部 279 个测试任务均已完成有效的初始先验评估和可追溯的接受终态，初始评估发生在生成修正之前。终止原因统计：${JSON.stringify(summary.window_reflection.termination_reasons)}。零修正是允许的有效结果；评估了 279 个任务不表示修改了 279 天。未遗留接口失败、响应截断或模式校验失败终态，历史异常与重试均保留审计。曾修复三日外生范围的适配和中断恢复问题，并归档失效过程产物；最终核验仅使用恢复后有效终态，没有替换任何冻结数据、先验、知识或测试规则。

验证包括 Python 22 项中 19 通过、3 项可选神经依赖跳过，LLM 策略 11 项通过，外生上下文及传输重试合计 4 项通过，中断恢复专项 1 项通过，类型检查及独立性检查通过。恢复专项覆盖初始评估失败后续跑，确认每个有效窗口在预测生成前重新完成初始评估。

LLM 为 deepseek-v4.1-flash，常规输出上限 32,000 tokens、逐窗生成上限 65,536，单次超时 600 秒，窗口并发 8。接口返回的 usage 合计 ${reported} tokens；另有 ${reserved} tokens 是未返回 usage 的请求预算预留，不能视为实际计费量。密钥从环境读取，不包含在交付中。

predictions/raw.csv、adjusted.csv、final.csv 对应三个输出；最终原始尺度预测读取 final_predicted_original 列，predicted 为标准化最终预测。规则调整 CSV 的真实值还原列存在最多 ${Math.max(...Object.values(summary.stage_actual_original_max_abs_roundoff).map(Number)).toFixed(6)} 源单位的 float32 舍入差异，各阶段标准化真实值已交叉核对；独立复算统一使用原始 CSV 的真实值。外层提供 summary.json、逐点对比 CSV、逐日指标、图像、实验配置、知识及核验日志。交付 manifest 按相对路径与 SHA-256 校验全部文件；源数据、提交先验、gap 预测、时间网格、真实值、最终尺度和初始评估顺序均已独立核验。
`
  const before = (store.usage() as { calls: number }).calls
  const config = store.get<any>("config"), signal = new AbortController().signal
  await new ForecastService(config, store, signal).finalReport(content)
  store.event("report.reconciled", { source: "independently_verified_yaguo_summary", additional_llm_calls: 0 })
  await buildDelivery(config, store, signal)
  if ((store.usage() as { calls: number }).calls !== before) throw new Error("Report reconciliation unexpectedly used LLM calls")
  await store.project()
  console.log(JSON.stringify({ status: "verified", delivery: path.join(store.root, "best-model"), additional_llm_calls: 0 }))
} finally { store?.close(); await unlock() }
