"""Independently verify and summarize a completed prompt_yaguo UserPrior run."""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,json,sqlite3
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
def read(path):return json.loads(Path(path).read_text())
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=ROOT/'workspaces/yaguo-userprior-20261104')
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/yaguo-userprior')
    args=parser.parse_args();forecast=args.workspace/'.forecast';bundle=forecast/'best-model'
    audit=read(args.output/'input-audit.json');manifest=read(bundle/'manifest.json');task=read(forecast/'task.json')
    with sqlite3.connect(f'file:{forecast}/runtime/state.sqlite?mode=ro',uri=True) as database:
        state=json.loads(database.execute("SELECT value FROM kv WHERE key='state'").fetchone()[0])
        assert state['lifecycle']=='succeeded'
        usage=[{'status':status,'calls':count,'provider_reported_tokens':reported,'reserved_tokens_for_unknown_usage':reserved}
               for status,count,reported,reserved in database.execute(
                   'SELECT status,COUNT(*),COALESCE(SUM(used),0),COALESCE(SUM(CASE WHEN used IS NULL THEN reserved ELSE 0 END),0) FROM calls GROUP BY status')]
    assert manifest['model']=='UserPrior'
    assert sha(forecast/'inputs/dataset.csv')==audit['truth_sha256']
    assert manifest['files']['model/prior.csv']==audit['prior_sha256']==sha(bundle/'model/prior.csv')
    assert task['exogenous_columns']==audit['exogenous_columns'] and task['exogenous_history_steps']==288
    assert (task['seq_len'],task['forecast_gap'],task['pred_len'],task['window_stride'])==(711,57,96,96)
    expected_tasks=pd.read_csv(args.output/'expected-tasks.csv')
    expected_tasks=expected_tasks.loc[expected_tasks.split=='test'].reset_index(drop=True)
    expected=pd.DatetimeIndex(np.concatenate([pd.date_range(day,periods=96,freq='15min').to_numpy() for day in expected_tasks.target_start]))
    assert len(expected_tasks)==279 and len(expected)==26784
    assert expected[0]==pd.Timestamp('2025-10-01') and expected[-1]==pd.Timestamp('2026-07-06 23:45')
    frames={'prior':pd.read_csv(bundle/'predictions/raw.csv'),'adacast_final':pd.read_csv(bundle/'predictions/final.csv')}
    if (bundle/'predictions/adjusted.csv').is_file():frames['adjusted']=pd.read_csv(bundle/'predictions/adjusted.csv')
    source=pd.read_csv(forecast/'inputs/dataset.csv',usecols=['date','OT'])
    prior_source=pd.read_csv(bundle/'model/prior.csv',usecols=['date','OT'])
    for frame in frames.values():
        assert pd.DatetimeIndex(pd.to_datetime(frame.timestamp,format='mixed')).equals(expected)
        assert frame.window_index.nunique()==279 and frame.groupby('window_index').size().eq(96).all()
        assert frame.groupby('window_index').horizon_index.apply(list).apply(lambda indices:indices==list(range(96))).all()
    raw=frames['prior'];final=frames['adacast_final'];indices=raw.row_index.to_numpy(dtype=int)
    for frame in frames.values():
        for column in ['window_index','horizon_index','row_index','origin']:
            np.testing.assert_array_equal(frame[column],raw[column])
        # Adjusted/refined CSVs inverse-transform float32 labels; raw retains source precision.
        np.testing.assert_allclose(frame.actual_original,raw.actual_original,rtol=0,atol=.01)
    np.testing.assert_allclose(raw.actual_original,source.OT.iloc[indices],rtol=0,atol=1e-6)
    np.testing.assert_allclose(raw.origin.groupby(raw.window_index).first(),expected_tasks.origin_index,rtol=0,atol=0)
    np.testing.assert_allclose(raw.row_index.groupby(raw.window_index).first(),expected_tasks.target_start_index,rtol=0,atol=0)
    np.testing.assert_allclose(raw.predicted_original,prior_source.OT.iloc[indices],rtol=0,atol=.01)
    comparison=pd.DataFrame({'timestamp':expected,'window_index':raw.window_index,'horizon_index':raw.horizon_index,'actual':raw.actual_original})
    for name in ['prior','adjusted','adacast_final']:
        if name not in frames:continue
        frame=frames[name]
        columns=['final_predicted_original','adjusted_predicted_original','predicted_original'] if name=='adacast_final' else ['adjusted_predicted_original','predicted_original'] if name=='adjusted' else ['predicted_original']
        comparison[name]=frame[next(column for column in columns if column in frame)]
    scale=read(bundle/'model/model.json')['scale']
    for frame in frames.values():
        np.testing.assert_allclose(frame.actual,(raw.actual_original-scale['mean'])/scale['std'],rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(comparison.adacast_final,final.predicted*scale['std']+scale['mean'],rtol=1e-7,atol=.01)
    gap=pd.read_csv(forecast/'runs'/manifest['test']['run_id']/'gap_predictions.csv')
    gap_indices=np.concatenate([np.arange(origin,origin+57) for origin in expected_tasks.origin_index])
    assert len(gap)==279*57 and gap.groupby('window_index').size().eq(57).all()
    assert pd.DatetimeIndex(pd.to_datetime(gap.timestamp,format='mixed')).equals(pd.DatetimeIndex(pd.to_datetime(source.date.iloc[gap_indices],format='mixed')))
    np.testing.assert_allclose(gap.predicted*scale['std']+scale['mean'],prior_source.OT.iloc[gap_indices],rtol=0,atol=.01)
    def metrics(frame,column):
        error=frame[column].to_numpy(dtype=float)-frame.actual.to_numpy(dtype=float)
        denominator=float(np.abs(frame.actual).sum())
        return {'mse':float(np.mean(error**2)),'rmse':float(np.sqrt(np.mean(error**2))),'mae':float(np.mean(abs(error))),
                'wape':float(np.sum(abs(error))/denominator) if denominator else None}
    names=[name for name in ['prior','adjusted','adacast_final'] if name in comparison]
    scores={name:metrics(comparison,name) for name in names}
    daily=pd.DataFrame([{'window_index':int(index),'date':str(day.timestamp.iloc[0].date()),'model':name,'points':len(day),**metrics(day,name)}
                        for index,day in comparison.groupby('window_index',sort=True) for name in names])
    mse=daily.pivot(index='date',columns='model',values='mse');gain=mse.prior-mse.adacast_final
    max_daily_delta=(comparison.adacast_final-comparison.prior).abs().groupby(comparison.timestamp.dt.strftime('%Y-%m-%d')).max()
    meaningful=max_daily_delta>.01
    largest=gain.idxmax();others=mse.drop(index=largest)
    regression_concentration=None
    if gain.min()<0:
        worst=gain.idxmin();without_worst=mse.drop(index=worst)
        regression_concentration={'largest_loss_day':str(worst),'day_prior_mse':float(mse.loc[worst,'prior']),
                                  'day_final_mse':float(mse.loc[worst,'adacast_final']),
                                  'largest_day_share_of_net_increase':float(gain.loc[worst]/gain.sum()) if gain.sum()<0 else None,
                                  'excluding_largest_loss_day':{'tasks':278,'prior_mse':float(without_worst.prior.mean()),
                                                               'final_mse':float(without_worst.adacast_final.mean()),
                                                               'mse_improvement_ratio':float(1-without_worst.adacast_final.mean()/without_worst.prior.mean())},
                                  'scope':'Post-hoc sensitivity only; both predictions omit the same day and all 279 days remain in the primary test'}
    decisions=read(bundle/'audit/decisions.json');record=next(r for r in decisions['domain']['reflection']['test_records'] if r['test_run_id']==manifest['test']['run_id'])
    root=bundle/'audit/reflection/test'/manifest['test']['run_id']
    terminals=[read(file) for file in root.glob('*/corrections/window-*.json')]
    failures={'provider_failed','schema_failed','response_empty','output_truncated'}
    for terminal in terminals:
        assert terminal['accepted']
        calls=[]
        for file in root.glob(f"*/corrections/llm-calls/window-{terminal['window_index']:06d}/call-*.json"):
            call=read(file)
            if 'call_type' in call:calls.append(call)
        initial_assessments=[call for call in calls if call['call_type']=='evidence_gap' and call['status']=='success'
                             and not call.get('request',{}).get('state',{}).get('evidences')]
        generations=[call for call in calls if call['call_type']=='unified_reasoning' and call['status']=='success']
        assert initial_assessments, 'Terminal window has no successful initial prior assessment'
        if generations:assert min(call['sequence'] for call in initial_assessments)<min(call['sequence'] for call in generations)
        loop=terminal.get('evidence_loop')
        if loop:
            assert loop['termination_reason'] not in failures
            if loop['termination_reason'] in {'evidence_exhausted','no_valid_transition'}:
                state=terminal['critic_prediction_state'];committed={item['evidence_candidate_id'] for item in state['evidences']}
                assert not any(item.get('reason') in failures and item.get('candidate_id') not in committed for item in loop['rejected_transitions'])
    checked_contexts=0
    for file in root.glob('*/blind-contexts/window-??????.json'):
        context=read(file)
        initial=read(file.with_name(file.stem+'.critic-prediction-state.initial.json'))
        anchor=raw.loc[raw.window_index==initial['window_index']]
        assert initial['evidences']==[] and len(initial['predictions'])==96
        np.testing.assert_array_equal([point['horizon_index'] for point in initial['predictions']],anchor.horizon_index)
        np.testing.assert_allclose([point['value'] for point in initial['predictions']],anchor.predicted_original,rtol=0,atol=.01)
        if 'lookback_window_exogenous_table' in context:
            assert context['visible_history_target_table']['row_count']==711
            assert context['lookback_window_exogenous_table']['row_count']==288
            assert context['predict_window_exogenous_table']['row_count']==96
            assert context['recent_three_day_exogenous_table']['row_count']==441
            for key in ['lookback_window_exogenous_table','predict_window_exogenous_table','recent_three_day_exogenous_table']:
                assert set(context[key]['columns'])=={'timestamp',*audit['exogenous_columns']}
        else:
            assert len(context['history_rows'])==711
            assert all(not row['known_exogenous'] for row in context['history_rows'][:-288])
            assert all(set(row['known_exogenous'])==set(audit['exogenous_columns']) for row in context['history_rows'][-288:])
        checked_contexts+=1
    assert len(terminals)>=record['corrected_count']
    if record.get('llm_refinement_enabled'):
        assert len(terminals)==record['llm_required_count']
    assert checked_contexts>=len(terminals)
    standard={'prior':manifest['test']['metrics'],'adacast_final':manifest['final_metrics']}
    validation_candidates=[]
    for file in (bundle/'audit/adjustments').glob('validation/*/*/adjusted_eval.json'):
        evaluation=read(file)
        validation_candidates.append({key:evaluation[key] for key in ['candidate_id','raw_metrics','adjusted_metrics','acceptance_decision'] if key in evaluation})
    if 'adjusted' in frames:
        application=next(r for r in decisions['domain']['adjustment']['test_application_records'] if r['test_run_id']==manifest['test']['run_id'] and r['decision']=='apply')
        standard['adjusted']=read(bundle/'audit'/application['adjusted_eval_path'].removeprefix('.forecast/'))['adjusted_metrics']
    for name,official in standard.items():
        np.testing.assert_allclose(scores[name]['mse']/scale['std']**2,official['mse'],rtol=1e-5,atol=1e-9)
        np.testing.assert_allclose(scores[name]['mae']/scale['std'],official['mae'],rtol=1e-5,atol=1e-9)
    summary={'protocol':{'source_prompt':'dataset/prompt_yaguo.md','training_points':157543,'adjustment_fit_tasks':45,'heldout_validation_tasks':45,
                         'test_tasks':279,'test_points':26784,'history_steps':711,'forecast_gap':57,'horizon_steps':96,'stride':96,
                         'test_start':str(expected[0]),'test_end':str(expected[-1]),'prediction_grid_anchor':audit['prediction_grid_anchor'],
                         'excluded_validation_days':['2025-08-15','2025-09-30'],'exclusion_reason':'Complete targets cannot cross the fit/holdout or validation/test origin boundary'},
             'provenance':{key:audit[key] for key in ['truth_path','prior_path','truth_sha256','prior_sha256']},
             'exogenous_scope':{'city':audit['selected_city'],'city_selection':audit['city_selection'],'columns':audit['exogenous_columns'],'history_steps':288,'future_steps':153,'checked_blind_contexts':checked_contexts},
             'full_test_original_scale':scores,'full_test_standardized_scale':standard,
             'validation':{'prior_holdout_metrics':manifest['validation']['metrics'],'candidate_evaluations':validation_candidates,'reflection_records':decisions['domain']['reflection']['validation_records']},
             'mse_improvement_ratio':float(1-scores['adacast_final']['mse']/scores['prior']['mse']),
             'mae_improvement_ratio':float(1-scores['adacast_final']['mae']/scores['prior']['mae']),
             'daily_mse_comparison':{'improved':int(((gain>1e-8)&meaningful).sum()),'worse':int(((gain< -1e-8)&meaningful).sum()),'unchanged':int(((abs(gain)<=1e-8)|~meaningful).sum()),
                                     'precision_rule':'Days with every prediction within 0.01 source units of the prior count as unchanged; primary aggregate metrics retain full precision'},
             'daily_mse_comparison_raw_numeric':{'improved':int((gain>1e-8).sum()),'worse':int((gain< -1e-8).sum()),'unchanged':int((abs(gain)<=1e-8).sum())},
             'gain_concentration':{'largest_gain_day':str(largest),'largest_day_share_of_net_reduction':float(gain.loc[largest]/gain.sum()) if gain.sum()!=0 else None,
                                   'excluding_largest_gain_day':{'tasks':278,'prior_mse':float(others.prior.mean()),'final_mse':float(others.adacast_final.mean()),'mse_improvement_ratio':float(1-others.adacast_final.mean()/others.prior.mean())},
                                   'scope':'Post-hoc sensitivity only; all 279 days remain in the primary test'},
             'regression_concentration':regression_concentration,
             'prior_replay_max_abs_original_difference':float(np.max(abs(raw.predicted_original.to_numpy()-prior_source.OT.iloc[indices].to_numpy()))),
             'stage_actual_original_max_abs_roundoff':{name:float(np.max(abs(frame.actual_original.to_numpy()-raw.actual_original.to_numpy()))) for name,frame in frames.items()},
             'actual_precision_check':'Raw labels match source within 1e-6 source units. Every stage matches training-normalized source labels; inverse-transformed float32 labels allow 0.01 source units of roundoff.',
             'window_reflection':{'numeric_anchor':'raw_submitted_prior','locked_skill_usage':'reference_only','record':record,'terminal_records':len(terminals),'termination_reasons':dict(Counter(r.get('evidence_loop',{}).get('termination_reason','legacy') for r in terminals)),'unresolved_request_failures':0},
             'llm_usage_by_status':usage,'usage_accounting':'Returned usage is provider-reported; unreturned usage reservations are estimates, not actual billing',
             'kpie':{'computed':False,'reason':'Business formula was not provided; MSE and MAE are the requested primary metrics'},
             'checks':{'exact_source_prior_only':True,'frozen_dataset_hash':True,'exact_daily_target_grid':True,'complete_test_points':True,'raw_prior_reused_with_float32_tolerance':True,'immutable_initial_states_match_raw_prior':True,'gap_forecasts_match_supplied_prior':True,'actuals_match_source':True,'stage_coordinates_match':True,'final_scale_consistent':True,'official_metrics_match_independent_recomputation':True,'exogenous_scope_checked':True,'initial_assessment_precedes_generation':True,'no_request_failure_terminal':True}}
    comparison.to_csv(args.output/'test-predictions.csv',index=False);daily.to_csv(args.output/'daily-test-metrics.csv',index=False)
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig,axes=plt.subplots(2,1,figsize=(14,7),constrained_layout=True)
    for name,label,color in [('actual','Actual','#64748b'),('prior','Submitted prior','#2563eb'),('adacast_final','AdaCast final','#c2410c')]:
        axes[0].plot(comparison.timestamp,comparison[name],linewidth=.5,alpha=.8,color=color,label=label)
    axes[0].set(title='Huabei | 279 daily tasks | 711 history + 57 gap -> 96 target steps',ylabel='Load (source units)');axes[0].legend(ncols=3,frameon=False)
    axes[1].bar(pd.to_datetime(gain.index),gain.values,color=['#0f766e' if value>=0 else '#b91c1c' for value in gain],width=.8)
    direction='lower' if summary['mse_improvement_ratio']>=0 else 'higher'
    axes[1].axhline(0,color='#475569',linewidth=.7);axes[1].set(ylabel='Prior MSE - final MSE',xlabel='Prediction day',title=f'Daily MSE change (positive = improvement); full-test MSE {abs(100*summary["mse_improvement_ratio"]):.2f}% {direction}')
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));ax.grid(axis='y',alpha=.15)
    fig.savefig(args.output/'test-comparison.png',dpi=180);fig.savefig(args.output/'test-comparison.svg');plt.close(fig)
    print(json.dumps({'original_scale_metrics':scores,'mse_improvement_ratio':summary['mse_improvement_ratio'],'summary':str(args.output/'summary.json')},indent=2))

if __name__=='__main__':main()
