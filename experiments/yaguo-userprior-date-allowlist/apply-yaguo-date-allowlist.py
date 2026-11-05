"""Replay completed Yaguo corrections only on user-allowlisted target days."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    error = np.asarray(predicted, dtype=float) - actual
    denominator = float(np.abs(actual).sum())
    return {'mse': float(np.mean(error**2)),
            'rmse': float(np.sqrt(np.mean(error**2))),
            'mae': float(np.mean(abs(error))),
            'wape': float(np.sum(abs(error)) / denominator) if denominator else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'deliveries/yaguo-userprior-20261104')
    parser.add_argument('--allowlist', type=Path, default=ROOT / 'examples/yaguo-test-correction-date-allowlist.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/yaguo-userprior-date-allowlist')
    parser.add_argument('--delivery', type=Path, default=ROOT / 'deliveries/yaguo-userprior-date-allowlist-20261104')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    delivery = args.delivery.resolve()
    assert output != source and not output.is_relative_to(source)
    assert delivery != source and not delivery.is_relative_to(source)
    assert not output.exists(), f'Output already exists: {output}'
    assert not delivery.exists(), f'Delivery already exists: {delivery}'

    source_index = read(source / 'result-index.json')
    source_summary = read(source / 'summary.json')
    assert source_index['status'] == 'succeeded' and all(source_summary['checks'].values())
    comparison_path = source / 'test-predictions.csv'
    assert sha(comparison_path) == source_index['files']['test-predictions.csv']
    assert sha(source / 'summary.json') == source_index['files']['summary.json']
    parent_manifest_path = source / 'adacast/manifest.json'
    assert sha(parent_manifest_path) == source_index['bundle']['manifest_sha256']
    parent_manifest = read(parent_manifest_path)
    assert parent_manifest['model'] == 'UserPrior'
    assert parent_manifest['files']['model/prior.csv'] == source_summary['provenance']['prior_sha256']
    for relative in ['predictions/raw.csv', 'model/model.json']:
        assert sha(source / 'adacast' / relative) == parent_manifest['files'][relative]

    allowed = read(args.allowlist)['TEST_CORRECTION_DATE_ALLOWLIST']
    assert allowed and len(allowed) == len(set(allowed)), 'Dates must be unique'
    assert all(isinstance(day, str) and pd.Timestamp(day).strftime('%Y-%m-%d') == day for day in allowed)
    frame = pd.read_csv(comparison_path)
    timestamps = pd.to_datetime(frame.timestamp, format='mixed')
    days = timestamps.dt.strftime('%Y-%m-%d')
    all_days = sorted(days.unique())
    assert set(allowed).issubset(all_days), 'Allowlist contains a day outside the test labels'
    assert len(frame) == 26784 and len(all_days) == 279
    assert timestamps.is_unique and timestamps.is_monotonic_increasing
    assert timestamps.iloc[0] == pd.Timestamp('2025-10-01')
    assert timestamps.iloc[-1] == pd.Timestamp('2026-07-06 23:45')
    assert frame.groupby(days).size().eq(96).all()
    assert frame.groupby('window_index').horizon_index.apply(list).apply(lambda values: values == list(range(96))).all()
    assert frame.groupby('window_index').timestamp.apply(lambda values: pd.to_datetime(values).dt.normalize().nunique()).eq(1).all()
    mask = days.isin(allowed).to_numpy()
    prior = frame.prior.to_numpy(dtype=float)
    unrestricted = frame.adacast_final.to_numpy(dtype=float)
    final = np.where(mask, unrestricted, prior)
    assert np.isfinite(frame.actual).all() and np.isfinite(final).all()
    assert mask.sum() == len(allowed) * 96
    np.testing.assert_array_equal(final[~mask], prior[~mask])
    np.testing.assert_array_equal(final[mask], unrestricted[mask])

    comparison = frame[['timestamp', 'window_index', 'horizon_index', 'actual', 'prior']].copy()
    comparison['adacast_unrestricted'] = unrestricted
    comparison['correction_allowed'] = mask
    comparison['adacast_final'] = final
    scores = {name: metrics(frame.actual, values) for name, values in
              [('prior', prior), ('adacast_unrestricted', unrestricted), ('adacast_date_allowlist', final)]}
    np.testing.assert_allclose(scores['prior']['mse'], source_summary['full_test_original_scale']['prior']['mse'], rtol=0, atol=1e-8)
    np.testing.assert_allclose(scores['adacast_unrestricted']['mse'], source_summary['full_test_original_scale']['adacast_final']['mse'], rtol=0, atol=1e-8)
    scale = read(source / 'adacast/model/model.json')['scale']
    standardized = {name: {'mse': score['mse'] / scale['std']**2, 'mae': score['mae'] / scale['std']}
                    for name, score in scores.items()}
    scope_scores = {}
    for scope, select in [('allowlisted_days', mask), ('other_days', ~mask)]:
        scope_scores[scope] = {'days': int(days.loc[select].nunique()), 'points': int(select.sum()),
                              'prior': metrics(frame.actual.loc[select], prior[select]),
                              'adacast_date_allowlist': metrics(frame.actual.loc[select], final[select])}
    daily = []
    for day, indices in frame.groupby(days, sort=True).groups.items():
        for name, values in [('prior', prior), ('adacast_unrestricted', unrestricted), ('adacast_date_allowlist', final)]:
            daily.append({'date': day, 'correction_allowed': day in allowed, 'model': name,
                          'points': len(indices), **metrics(frame.actual.iloc[indices], values[indices])})
    daily = pd.DataFrame(daily)
    daily_mse = daily.pivot(index='date', columns='model', values='mse')
    gain = daily_mse.prior - daily_mse.adacast_date_allowlist
    changed = (comparison.adacast_final - comparison.prior).abs().groupby(days).max() > .01
    assert not changed.loc[~changed.index.isin(allowed)].any()

    raw = pd.read_csv(source / 'adacast/predictions/raw.csv')
    np.testing.assert_array_equal(raw.window_index, frame.window_index)
    np.testing.assert_array_equal(raw.horizon_index, frame.horizon_index)
    assert pd.to_datetime(raw.timestamp, format='mixed').equals(timestamps)
    predictions = raw[['window_index', 'origin', 'row_index', 'horizon_index', 'timestamp']].copy()
    predictions['actual'] = (frame.actual - scale['mean']) / scale['std']
    predictions['predicted'] = (final - scale['mean']) / scale['std']
    predictions['actual_original'] = frame.actual
    predictions['prior_predicted_original'] = prior
    predictions['predicted_original'] = final
    predictions['final_predicted_original'] = final
    predictions['correction_allowed'] = mask

    policy = {'mode': 'user_date_allowlist_replay', 'date_field': 'prediction-label local calendar date',
              'TEST_CORRECTION_DATE_ALLOWLIST': allowed,
              'allowed_action': 'Reuse the previously completed AdaCast final predictions for this date',
              'other_action': 'Use the prior column from the verified original UserPrior replay point for point',
              'source': str(source), 'additional_api_calls': 0}
    checks = {'source_artifact_hashes_verified': True, 'unique_valid_allowlist_dates': True,
              'all_allowlist_dates_in_test': True, 'all_279_days_retained': True,
              'all_days_have_96_points': True, 'allowed_dates_match_previous_adacast': True,
              'other_dates_equal_prior_exactly': True, 'window_coordinates_unchanged': True,
              'actuals_unchanged': True, 'csv_roundtrip_preserves_policy': True}
    summary = {'status': 'succeeded', 'mode': policy['mode'], 'additional_api_calls': 0,
               'protocol': {**source_summary['protocol'], 'allowlisted_test_days': len(allowed),
                            'allowlisted_points': int(mask.sum()), 'prior_only_days': len(all_days) - len(allowed),
                            'prior_only_points': int((~mask).sum()), 'date_matching': policy['date_field']},
               'TEST_CORRECTION_DATE_ALLOWLIST': allowed,
               'provenance': {**source_summary['provenance'], 'parent_delivery': str(source),
                              'parent_manifest_sha256': sha(parent_manifest_path),
                              'parent_prediction_csv_sha256': sha(comparison_path)},
               'exogenous_scope': source_summary['exogenous_scope'],
               'full_test_original_scale': scores, 'full_test_standardized_scale': standardized,
               'scope_metrics_original_scale': scope_scores,
               'mse_change_ratio_vs_prior': scores['adacast_date_allowlist']['mse'] / scores['prior']['mse'] - 1,
               'mae_change_ratio_vs_prior': scores['adacast_date_allowlist']['mae'] / scores['prior']['mae'] - 1,
               'daily_mse_comparison': {'improved': int(((gain > 1e-8) & changed).sum()),
                                        'worse': int(((gain < -1e-8) & changed).sum()),
                                        'unchanged': int(((abs(gain) <= 1e-8) | ~changed).sum()),
                                        'meaningful_change_tolerance_original': .01},
               'allowlisted_changed_days': int(changed.loc[changed.index.isin(allowed)].sum()),
               'prior_semantics': 'Same prior output as the previous verified run; its original source file and normalization-roundoff provenance are unchanged.',
               'checks': checks}
    output.mkdir(parents=True)
    (output / 'predictions').mkdir()
    comparison.to_csv(output / 'test-predictions.csv', index=False)
    predictions.to_csv(output / 'predictions/final.csv', index=False)
    daily.to_csv(output / 'daily-test-metrics.csv', index=False)
    persisted = pd.read_csv(output / 'test-predictions.csv', float_precision='round_trip')
    np.testing.assert_array_equal(persisted.adacast_final.to_numpy()[~mask], prior[~mask])
    np.testing.assert_array_equal(persisted.adacast_final.to_numpy()[mask], unrestricted[mask])
    persisted_final = pd.read_csv(output / 'predictions/final.csv', float_precision='round_trip')
    np.testing.assert_array_equal(persisted_final.final_predicted_original, final)
    np.testing.assert_allclose(persisted_final.predicted * scale['std'] + scale['mean'], final, rtol=0, atol=1e-8)
    for name, value in [('policy.json', policy), ('summary.json', summary)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    shutil.copy2(args.allowlist, output / 'test-correction-date-allowlist.json')
    shutil.copy2(Path(__file__), output / Path(__file__).name)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True)
    for values, label, color in [(frame.actual, 'Actual', '#64748b'), (prior, 'Prior', '#2563eb'), (final, 'AdaCast: date allowlist', '#c2410c')]:
        axes[0].plot(timestamps, values, linewidth=.5, alpha=.8, color=color, label=label)
    axes[0].set(title=f'Huabei | {len(allowed)} AdaCast dates + {len(all_days)-len(allowed)} prior-only dates', ylabel='Load (source units)')
    axes[0].legend(ncols=3, frameon=False)
    axes[1].bar(pd.to_datetime(gain.index), gain.values, width=.8, color=['#0f766e' if v >= 0 else '#b91c1c' for v in gain])
    change = summary['mse_change_ratio_vs_prior']
    axes[1].set(title=f'Full 279-day MSE {abs(change)*100:.2f}% {"higher" if change >= 0 else "lower"} than prior; unlisted dates unchanged',
                ylabel='Prior MSE - allowlisted MSE', xlabel='Prediction day')
    axes[1].axhline(0, linewidth=.7, color='#475569')
    for axis in axes:
        axis.xaxis.set_major_locator(mdates.MonthLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axis.grid(axis='y', alpha=.15)
    fig.savefig(output / 'test-comparison.png', dpi=180)
    fig.savefig(output / 'test-comparison.svg')
    plt.close(fig)
    rows = '\n'.join(f"| {label} | {scores[name]['mse']:,.6f} | {scores[name]['mae']:,.6f} |" for name, label in
                     [('prior', '原始 prior'), ('adacast_unrestricted', '上轮全日期 AdaCast'), ('adacast_date_allowlist', '本轮白名单 AdaCast')])
    mse_direction = '上升' if change >= 0 else '下降'
    mae_change = summary['mae_change_ratio_vs_prior']
    mae_direction = '上升' if mae_change >= 0 else '下降'
    report = f'''# Yaguo：按用户日期白名单应用修正

本次按用户提供的 TEST_CORRECTION_DATE_ALLOWLIST 对已完成结果进行日期控制回放。按预测标签的本地日历日期匹配，共 {len(allowed)} 天、{int(mask.sum()):,} 点复用上轮 AdaCast 最终修正；其余 {len(all_days)-len(allowed)} 天、{int((~mask).sum()):,} 点逐点使用上轮原始 prior 输出。完整测试仍为 2025-10-01 至 2026-07-06 的 279 天、26,784 点。本轮新增 API 调用为 0。

## 完整测试指标（原始负荷尺度）

| 输出 | MSE | MAE |
| --- | ---: | ---: |
{rows}

本轮相对原始 prior：MSE {mse_direction} {abs(change)*100:.2f}%，MAE {mae_direction} {abs(mae_change)*100:.2f}%。逐日 MSE 为 {summary['daily_mse_comparison']['improved']} 天改善、{summary['daily_mse_comparison']['worse']} 天变差、{summary['daily_mse_comparison']['unchanged']} 天不变。白名单内有 {summary['allowlisted_changed_days']} 天的预测相对 prior 存在超过 0.01 源单位的数值变化；白名单外所有预测与 prior 严格相等。

prior 使用与上一轮比较表相同的原始模型输出，其源文件仍为实验设置指定的 prediction_prior.csv（SHA-256：{source_summary['provenance']['prior_sha256']}），保留上一轮标准化及还原的数值精度。训练/验证切分、711 点负荷历史、57 点 gap、96 点预测、stride=96，以及秦皇岛市和最近 288 点历史外生范围沿用上一轮设置。此报告是已完成实验的白名单回放结果。

## 文件与核验

- test-predictions.csv：真实值、prior、上轮全日期输出、本轮输出及 correction_allowed 标志。
- predictions/final.csv：带完整窗口坐标的本轮最终预测；原始尺度读取 final_predicted_original，标准化尺度读取 predicted。
- daily-test-metrics.csv：三个输出的逐日指标；summary.json 另列白名单内与白名单外子集指标。
- test-correction-date-allowlist.json、policy.json：本轮白名单及执行语义。
- test-comparison.png / .svg：完整轨迹及逐日 MSE 差值。
- manifest.json：本目录文件的 SHA-256；原实验与全部 LLM 审计保存在 {source}。

已检查原结果及来源哈希、完整 279 天、每日 96 点、窗口坐标、真实值、白名单内逐点复用、白名单外逐点 prior，以及 CSV 写入后的数值一致性。全部 {len(checks)} 项检查通过。
'''
    (output / 'report.md').write_text(report)
    files = {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob('*')) if path.is_file()}
    manifest = {'schema_version': 1, 'status': 'succeeded', 'mode': policy['mode'],
                'parent_manifest_sha256': sha(parent_manifest_path), 'final_prediction': 'predictions/final.csv',
                'test_tasks': 279, 'allowlisted_days': len(allowed), 'prior_only_days': len(all_days)-len(allowed),
                'additional_api_calls': 0, 'all_checks_passed': True, 'files': files}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    for relative, digest in files.items():
        assert sha(output / relative) == digest
    shutil.copytree(output, delivery)
    assert sha(output / 'manifest.json') == sha(delivery / 'manifest.json')
    for relative, digest in files.items():
        assert sha(delivery / relative) == digest
    print(json.dumps({'status': 'succeeded', 'allowlisted_days': len(allowed), 'prior_only_days': len(all_days)-len(allowed),
                      'full_test_original_scale': scores, 'mse_change_ratio_vs_prior': change,
                      'mae_change_ratio_vs_prior': mae_change, 'daily_mse_comparison': summary['daily_mse_comparison'],
                      'delivery': str(delivery), 'checks_passed': len(checks)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
