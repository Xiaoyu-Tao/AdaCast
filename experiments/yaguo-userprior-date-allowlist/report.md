# Yaguo：按用户日期白名单应用修正

本次按用户提供的 TEST_CORRECTION_DATE_ALLOWLIST 对已完成结果进行日期控制回放。按预测标签的本地日历日期匹配，共 60 天、5,760 点复用上轮 AdaCast 最终修正；其余 219 天、21,024 点逐点使用上轮原始 prior 输出。完整测试仍为 2025-10-01 至 2026-07-06 的 279 天、26,784 点。本轮新增 API 调用为 0。

## 完整测试指标（原始负荷尺度）

| 输出 | MSE | MAE |
| --- | ---: | ---: |
| 原始 prior | 771,930.675941 | 663.867903 |
| 上轮全日期 AdaCast | 997,610.401786 | 726.791511 |
| 本轮白名单 AdaCast | 815,705.394156 | 672.524518 |

本轮相对原始 prior：MSE 上升 5.67%，MAE 上升 1.30%。逐日 MSE 为 23 天改善、27 天变差、229 天不变。白名单内有 50 天的预测相对 prior 存在超过 0.01 源单位的数值变化；白名单外所有预测与 prior 严格相等。

prior 使用与上一轮比较表相同的原始模型输出，其源文件仍为实验设置指定的 prediction_prior.csv（SHA-256：608b4da2683a119917fe374d9f3228e9957601cb34ebcc65e90a23f47eb3c4c8），保留上一轮标准化及还原的数值精度。训练/验证切分、711 点负荷历史、57 点 gap、96 点预测、stride=96，以及秦皇岛市和最近 288 点历史外生范围沿用上一轮设置。此报告是已完成实验的白名单回放结果。

## 文件与核验

- test-predictions.csv：真实值、prior、上轮全日期输出、本轮输出及 correction_allowed 标志。
- predictions/final.csv：带完整窗口坐标的本轮最终预测；原始尺度读取 final_predicted_original，标准化尺度读取 predicted。
- daily-test-metrics.csv：三个输出的逐日指标；summary.json 另列白名单内与白名单外子集指标。
- test-correction-date-allowlist.json、policy.json：本轮白名单及执行语义。
- test-comparison.png / .svg：完整轨迹及逐日 MSE 差值。
- manifest.json：本目录文件的 SHA-256；原实验与全部 LLM 审计保存在 /mnt/d/kawai10726/work/codes/project/AdaCast/adacast-lite/deliveries/yaguo-userprior-20261104。

已检查原结果及来源哈希、完整 279 天、每日 96 点、窗口坐标、真实值、白名单内逐点复用、白名单外逐点 prior，以及 CSV 写入后的数值一致性。全部 10 项检查通过。
