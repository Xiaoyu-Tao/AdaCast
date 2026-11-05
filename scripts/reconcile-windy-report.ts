import path from "node:path"
import {Store,lockWorkspace} from "../src/storage"
import {ForecastService} from "../src/forecast"
import {buildDelivery,verifyBundle} from "../src/delivery"
import {canonical,readJSON} from "../src/util"

// Run after summarize-windy-chronos2.py has independently verified the completed run.
const root=path.resolve(import.meta.dir,"..")
const workspace=path.join(root,"workspaces/windy-chronos2-ms-visible-20261103")
const summary=await readJSON(path.join(root,"experiments/windy-chronos2-ms-visible/summary.json"))
if(!Object.values(summary.checks).every(value=>value===true))throw new Error("Independent verification is incomplete")
const unlock=await lockWorkspace(workspace)
let store:Store|undefined
try {
  store=new Store(path.join(workspace,".forecast"))
  if(store.get<any>("state")?.lifecycle!=="succeeded")throw new Error("Complete the workflow before reconciling its report")
  const old=await verifyBundle(path.join(store.root,"best-model"))
  if(canonical(old.final_metrics)!==canonical(summary.full_test_standardized_scale.adacast_final))throw new Error("Summary metrics differ from completed delivery")
  const original=summary.full_test_original_scale
  const rows=[["Chronos-2 原始先验","chronos2"],["锁定规则调整","adjusted"],["AdaCast 最终输出","adacast_final"]]
    .map(([label,key])=>`| ${label} | ${original[key].mse.toFixed(6)} | ${original[key].mae.toFixed(6)} | ${(100*original[key].wape).toFixed(4)}% |`).join("\n")
  const counts=summary.daily_mse_comparison
  const concentration=summary.gain_concentration
  const usage=summary.llm_usage_by_status
  const reported=usage.reduce((sum:number,row:any)=>sum+row.provider_reported_tokens,0)
  const reserved=usage.reduce((sum:number,row:any)=>sum+row.reserved_tokens_for_unknown_usage,0)
  const maeGain=100*(1-original.adacast_final.mae/original.chronos2.mae)
  const content=`实验严格采用每日滑动的 96 点历史 → 96 点预测，stride=96、gap=0，预测标签不重叠，只预测 real_power。训练段为 2024-01-01 00:00 至 2024-09-02 18:15，共 23,594 点；其后 22 点仅作为历史。验证标签为 9 月 3 日至 11 月 11 日，共 70 天、6,720 点；测试标签为 11 月 12 日至 12 月 31 日，共 50 天、4,800 点。第一个测试任务输入 11 月 11 日，最后一个输入 12 月 30 日。

Chronos-2 使用 ${summary.chronos2.model}，固定 revision ${summary.chronos2.revision}，以 CUDA bfloat16 零样本推理。54 个预测日气象外生变量可见，预测日 real_power 不进入模型输入或盲评修正输入。全空的未命名列不作为特征。原始 Chronos-2 预测作为唯一 UserPrior，未执行模型库搜索或微调。

验证集前 35 天供规则拟合和校准，后 35 天供独立验收。完整 70 天原始先验验证 MSE 为 ${summary.chronos2.full_validation_metrics.mse_original}（原始尺度）。上方自动表中的 Validation 指后 35 天，不能与完整验证集混用。轻量规则候选 C 经既有 MAE/WAPE 验收标准接受并锁定为 v001；它并未降低验证 MSE。原始先验及规则调整后的两组窗口反思均通过验证阶段的 MSE 门槛，随后执行完整测试窗口反思。

测试集原始尺度指标：

| 阶段 | MSE | MAE | WAPE |
| --- | ---: | ---: | ---: |
${rows}

最终测试 MSE 较 Chronos-2 原始先验下降 ${(100*summary.final_mse_improvement_ratio).toFixed(2)}%，MAE 下降 ${maeGain.toFixed(2)}%。单独规则调整的测试 MSE 略有上升，最终改善来自后续窗口反思。逐日 MSE 为 ${counts.improved} 天改善、${counts.worse} 天变差、${counts.unchanged} 天不变，整体改善不代表每天均改善。原始尺度 WAPE 使用 sum(abs(error))/sum(abs(actual))；上方自动表中的 WAPE 使用标准化后的目标，二者口径不同。全零真实值日期的逐日 WAPE 记为空值。

改善高度集中于 ${concentration.largest_gain_day}：该日贡献全部净 MSE 降幅的 ${(100*concentration.largest_day_share_of_net_mse_reduction).toFixed(2)}%。仅作为事后敏感性分析，剔除该日后的其余 49 天，Chronos-2 先验 MSE 为 ${concentration.excluding_largest_gain_day.chronos2_mse.toFixed(6)}，AdaCast 最终 MSE 为 ${concentration.excluding_largest_gain_day.adacast_final_mse.toFixed(6)}，反而上升 ${(-100*concentration.excluding_largest_gain_day.mse_improvement_ratio).toFixed(2)}%。主要测试结果仍保留全部 50 天，未通过删日改变实验口径；本次结果不能解释为稳定的逐日改善。

所有 50 个测试窗口均具有有效接受终态，终态统计为 ${JSON.stringify(summary.window_reflection.termination_reasons)}，未遗留接口失败终态。历史接口异常及重试保留在审计记录中；失败请求不会被当作有效预测终态。完整 Bun 套件 33 项通过，状态同步更新后的恢复回归测试另有 3 项通过；Python 套件运行 20 项，其中 3 项因可选依赖跳过，类型检查通过。

常规 LLM 输出上限为 32,000 tokens，窗口预测生成上限为 65,536，单次请求超时 600 秒。当前服务地址为 https://xin.kabuai.cn/v1，模型为 deepseek-v4.1-flash；密钥通过环境变量读取。接口返回的 usage 共 ${reported} tokens；另有 ${reserved} tokens 为失败或中断请求的预算预留估计，不能视为实际计费量。

交付包中的 predictions/raw.csv、predictions/adjusted.csv、predictions/final.csv 分别对应三个阶段。最终 CSV 的原始尺度结果应读取 final_predicted_original 列，predicted 列为标准化最终结果。audit/reflection/、audit/adjustments/、audit/llm-calls/ 保存控制器、规则和调用记录；manifest.json 按相对路径及 SHA-256 校验全部交付文件。独立汇总、逐点对比 CSV、逐日指标和图像随外层交付目录一并提供。所有测试时间戳、源数据真实值、先验复用及最终尺度还原均经独立核验。
`
  const config=store.get<any>("config"),signal=new AbortController().signal
  const before=(store.usage() as {calls:number}).calls
  await new ForecastService(config,store,signal).finalReport(content)
  store.event("report.reconciled",{source:"independently_verified_summary",reason:"Align metric scopes, usage accounting and delivery paths with verified artifacts",additional_llm_calls:0})
  await buildDelivery(config,store,signal)
  if((store.usage() as {calls:number}).calls!==before)throw new Error("Report reconciliation unexpectedly used LLM calls")
  await store.project()
  console.log(JSON.stringify({status:"verified",delivery:path.join(store.root,"best-model"),additional_llm_calls:0}))
} finally {store?.close();await unlock()}
