import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from adacast_lite_ml.engine.adjustment.apply import safe_exogenous_columns, safe_text_exogenous_columns
from adacast_lite_ml.engine.reflection.controller import build_window_context
from adacast_lite_ml.runner import dispatch


class ExogenousScopeTests(unittest.TestCase):
    def test_explicit_daily_prior_grid_is_not_shifted_by_changed_gap_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            dates=pd.date_range('2025-01-01',periods=4000,freq='15min')
            original=pd.DataFrame({'date':dates.astype(str),'OT':100+np.sin(np.arange(4000)/10)})
            prior=original.copy();prior.loc[1383:,'OT']+=4
            original.to_csv(root/'data.csv',index=False);prior.to_csv(root/'prior.csv',index=False)
            task={'dataset_path':str(root/'data.csv'),'time_col':'date','target_col':'OT','frequency':'15min','seq_len':711,'label_len':711,
                  'pred_len':96,'forecast_gap':57,'window_stride':96,'split_mode':'timestamp',
                  'train_end_timestamp':str(dates[1382]),'validation_end_timestamp':str(dates[2342]),
                  'prediction_target_start_timestamp':str(dates[1440]),'model_families':['UserPrior']}
            source=None
            for split,expected in [('val','2025-01-21'),('adjustment_fit','2025-01-16'),('test','2025-01-26')]:
                request={'protocol_version':1,'operation':'experiment','task':task,'spec':{'model':'UserPrior','params':{}},'split':split,
                         'run_id':split,'output_dir':str(root/split)}
                if source:request.update(source_artifact=source['artifact'],source_sha256=source['artifact_sha256'],source_run_id='val')
                else:request['prior_path']=str(root/'prior.csv')
                result=dispatch(request)
                if source is None:source=result
                points=pd.read_csv(result['predictions'])
                self.assertEqual(pd.Timestamp(points.timestamp.iloc[0]),pd.Timestamp(expected))
                self.assertEqual(result['spec']['prediction_grid_target_anchor_index'],1440)
                self.assertTrue(pd.to_datetime(points.groupby('window_index').timestamp.first()).dt.hour.eq(0).all())

    def test_city_allowlist_and_three_day_history_preserve_target_history_and_hide_future_target(self):
        timestamps=pd.date_range('2025-06-20',periods=1200,freq='15min')
        dataset=pd.DataFrame({'date':timestamps.astype(str),'OT':np.arange(1200,dtype=float),
                              '秦皇岛市_t2m':np.arange(1200,dtype=float)/100,
                              '北京市_t2m':999.,'节假日':['ordinary']*1200})
        origin=900;target_start=origin+57
        task={'target_col':'OT','time_col':'date','seq_len':711,'pred_len':96,'forecast_gap':57,
              'exogenous_columns':['秦皇岛市_t2m','节假日'],'exogenous_history_steps':288}
        rows=pd.DataFrame({'window_index':0,'horizon_index':range(96),'timestamp':dataset.date.iloc[target_start:target_start+96].tolist()})
        gap=pd.DataFrame({'gap_index':range(57),'timestamp':dataset.date.iloc[origin:target_start].tolist(),'predicted_original':1.})
        policy={'constraints':{'absolute_min':0.,'absolute_max':10000.,'max_abs_step':1000.}}
        self.assertEqual(safe_exogenous_columns(task,dataset),['秦皇岛市_t2m'])
        self.assertEqual(safe_text_exogenous_columns(task,dataset),['节假日'])
        with tempfile.TemporaryDirectory() as directory:
            def context(frame):
                return build_window_context(task,Path(directory)/'task.json','2',frame,rows,np.ones(96),np.ones(96),{},[],
                    reasoning_mode='evidence_loop_test',policy=policy,condition_feature_rows=pd.DataFrame({'horizon_index':range(96)}),gap_prediction_rows=gap)
            before=context(dataset)
            changed=dataset.copy();changed.loc[origin:,'OT']=999999999.
            self.assertEqual(context(changed),before)
        self.assertEqual(before['visible_history_target_table']['row_count'],711)
        history=before['lookback_window_exogenous_table']
        self.assertEqual(history['row_count'],288)
        self.assertEqual(history['rows'][0][0],dataset.date.iloc[origin-288])
        self.assertEqual(history['rows'][-1][0],dataset.date.iloc[origin-1])
        self.assertEqual(before['predict_window_exogenous_table']['row_count'],96)
        self.assertEqual(before['recent_three_day_exogenous_table']['row_count'],288+57+96)
        for key in ['lookback_window_exogenous_table','predict_window_exogenous_table','recent_three_day_exogenous_table']:
            self.assertEqual(before[key]['columns'],['timestamp','秦皇岛市_t2m','节假日'])
        self.assertEqual(before['context_representation']['visible_target_history_rows'],711)


if __name__=='__main__':unittest.main()
