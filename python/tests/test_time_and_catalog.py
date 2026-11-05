"""Calendar, irregular-grid and independent model-install contracts."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from adacast_lite_ml.data import Data
from adacast_lite_ml.runner import dispatch
from adacast_lite_ml.catalog import catalog,canonical_model,preflight
from adacast_lite_ml.engine.utils.timefeatures import time_features


class CalendarAndCatalogTests(unittest.TestCase):
    def make(self,root,freq="h",start="2000-01-01",irregular=False):
        dates=pd.date_range(start,periods=360,freq=freq)
        if irregular:dates=dates.delete(70).append(pd.DatetimeIndex([dates[-1]+pd.Timedelta(hours=1)]))
        file=Path(root)/"series.csv"
        with file.open("w",newline="") as f:
            w=csv.writer(f);w.writerow(["date","target","weather"])
            for i,t in enumerate(dates):w.writerow([t.isoformat(),10+math.sin(i/4)+i/100,5+math.cos(i/7)])
        return {"dataset_path":str(file),"time_col":"date","target_col":"target","frequency":freq,"seq_len":24,"label_len":12,"pred_len":12,"forecast_gap":2,"window_stride":6,"train_ratio":.6,"val_ratio":.2,"test_ratio":.2,"model_families":["Naive"]}

    def test_calendar_frequencies_keep_observed_rows_and_normalized_values(self):
        for freq,start in [("W-SUN","2000-01-02"),("ME","2000-01-31"),("MS","2000-01-01"),("QE","1900-03-31"),("B","2000-01-03"),("YE","1800-12-31"),("15min","2000-01-01")]:
            with self.subTest(freq=freq),tempfile.TemporaryDirectory() as root:
                task=self.make(root,freq,start)
                data=Data(task)
                self.assertTrue(data.regular)
                result=dispatch({"protocol_version":1,"operation":"experiment","task":task,"spec":{"model":"Naive","params":{}},"split":"val","run_id":"val","output_dir":str(Path(root)/"val")})
                rows=pd.read_csv(result["predictions"])
                self.assertEqual(len(rows),len(data.origins("val"))*12)
                self.assertEqual(pd.Timestamp(rows.iloc[0].timestamp),pd.Timestamp(data.times[data.a+2]))
                self.assertTrue(np.isfinite(time_features(pd.DatetimeIndex(pd.to_datetime(data.times)),freq)).all())

    def test_irregular_series_uses_row_grid_and_strict_mode_is_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            task=self.make(root,irregular=True)
            data=Data(task)
            self.assertFalse(data.regular)
            self.assertEqual(list(data.origins("val")),[252,258,264,270])
            with self.assertRaisesRegex(ValueError,"strict mode"):Data({**task,"strict_frequency":True})

    def test_history_only_bridge_preserves_complete_daily_prior_windows(self):
        from adacast_lite_ml.engine.data_loader import resolve_training_boundaries, load_dataset
        from adacast_lite_ml.engine.reflection.controller import finite_policy_target_partitions
        with tempfile.TemporaryDirectory() as root:
            dates = pd.date_range("2024-01-01", periods=960, freq="15min")
            frame = pd.DataFrame({"date": dates.astype(str), "target": 10 + np.sin(np.arange(960)/10), "weather": np.arange(960)})
            file = Path(root)/"series.csv"
            frame.to_csv(file, index=False)
            task = {"dataset_path": str(file), "time_col": "date", "target_col": "target", "frequency": "15min",
                    "seq_len": 96, "label_len": 48, "pred_len": 96, "forecast_gap": 0, "window_stride": 96,
                    "split_mode": "timestamp", "train_end_timestamp": str(dates[375]),
                    "validation_start_timestamp": str(dates[384]), "validation_end_timestamp": str(dates[767]),
                    "model_families": ["Naive"]}
            data = Data(task)
            self.assertEqual(data.boundaries(), {"train": [0,376], "adjustment_fit": [384,576], "val": [576,768], "test": [768,960]})
            self.assertEqual(resolve_training_boundaries(frame, task), (376,576,768))
            self.assertEqual(list(data.origins("adjustment_fit")), [384,480])
            self.assertEqual(list(data.origins("val")), [576,672])
            self.assertEqual(list(data.origins("test")), [768,864])
            # Engine calibration and its model loader must agree with the API data grid.
            training, front = finite_policy_target_partitions(task, Path(root)/"task.json")
            self.assertEqual(len(training), 376)
            np.testing.assert_allclose(front, frame.target.iloc[384:576].to_numpy(), rtol=0, atol=1e-12)
            payload = load_dataset(task, ["adjustment_fit"])["adjustment_fit"]
            self.assertEqual(payload["absolute_start_index"], 288)
            self.assertEqual(payload["forecast_origin_index"], 96)
            self.assertEqual(payload["forecast_offset"], 8)
            request = {"protocol_version": 1, "operation_id": "bridge-test", "task": task}
            analysis = dispatch({**request, "operation": "analyze"})
            self.assertEqual(analysis["analysis_scope"]["target_row_count"], 376)
            self.assertEqual(analysis["analysis_scope"]["reserved_adjustment_row_count"], 192)
            prior = frame.copy()
            prior.loc[384:, "target"] += 0.5
            prior_file = Path(root)/"prior.csv"
            prior.to_csv(prior_file, index=False)
            fitted = dispatch({**request, "operation": "experiment", "spec": {"model": "UserPrior", "params": {}},
                               "split": "val", "run_id": "prior", "prior_path": str(prior_file), "output_dir": str(Path(root)/"prior")})
            self.assertEqual((fitted["windows"], fitted["points"]), (2,192))
            for split, first in [("adjustment_fit",384),("test",768)]:
                replayed = dispatch({**request, "operation": "experiment", "spec": fitted["spec"], "split": split,
                                     "run_id": split, "output_dir": str(Path(root)/split), "source_artifact": fitted["artifact"],
                                     "source_sha256": fitted["artifact_sha256"], "source_run_id": "prior"})
                points = pd.read_csv(replayed["predictions"])
                self.assertEqual((replayed["windows"],len(points)), (2,192))
                self.assertEqual(int(points.iloc[0].origin), first)
            for start in [str(dates[374]),str(dates[768])]:
                invalid = {**task,"validation_start_timestamp":start}
                with self.assertRaisesRegex(ValueError,"Validation must start"):
                    Data(invalid)
                with self.assertRaisesRegex(ValueError,"Validation must start"):
                    resolve_training_boundaries(frame, invalid)

    def test_minute_and_month_aliases_have_distinct_reference_semantics(self):
        from adacast_lite_ml.frequencies import normalize_frequency
        self.assertEqual(normalize_frequency("m"),"ME")
        self.assertEqual(normalize_frequency("t"),"min")
        self.assertEqual(normalize_frequency("15min"),"15min")
        self.assertEqual(normalize_frequency("2h"),"2h")

    def test_catalog_requires_neither_dataset_nor_foundation_weight_download(self):
        with patch("adacast_lite_ml.catalog.importlib.util.find_spec",return_value=None):
            result=dispatch({"protocol_version":1,"operation":"models"})
            rows={m["name"]:m for m in result["models"]}
            self.assertTrue(rows["Holt"]["dependencies_available"])
            self.assertFalse(rows["Chronos"]["dependencies_available"])
            self.assertEqual(rows["Moirai"]["install_extra"],"moirai")
            self.assertIn("mamba-ssm",rows["Mamba"]["missing_dependencies"])
            with self.assertRaisesRegex(ValueError,"extra 'timesfm'"):preflight(["TimesFM"])

    def test_model_aliases_and_catalog_match_independent_runtime_registry(self):
        self.assertEqual(canonical_model("holt-winters"),"HoltWinters")
        self.assertEqual(canonical_model("chronos2"),"Chronos2")
        self.assertEqual(canonical_model("dlinear"),"DLinear")
        names={m["name"] for m in catalog()["models"]}
        self.assertTrue({"DLinear","Transformer","Crossformer","Chronos2","TiRex"}<=names)

    @unittest.skipUnless(importlib.util.find_spec("torch"),"neural extra is not installed")
    def test_calendar_aware_transformer_runs_with_monthly_feature_width(self):
        with tempfile.TemporaryDirectory() as root:
            task=self.make(root,"ME","2000-01-31")
            result=dispatch({"protocol_version":1,"operation":"experiment","task":task,"spec":{"model":"Transformer","params":{"train_epochs":1,"d_model":16,"n_heads":2,"e_layers":1,"d_layers":1,"d_ff":32,"device":"cpu"}},"split":"val","run_id":"val","output_dir":str(Path(root)/"val")})
            self.assertTrue(math.isfinite(result["metrics"]["mse"]))

    @unittest.skipUnless(importlib.util.find_spec("torch"),"neural extra is not installed")
    def test_crossformer_imports_only_the_owned_model_namespace(self):
        from adacast_lite_ml.engine.models.Crossformer import Model
        from adacast_lite_ml.engine.runner import build_model_config
        config=build_model_config({"model":"Crossformer","seq_len":24,"label_len":12,"pred_len":12,"hyperparams":{"d_model":16,"n_heads":2,"e_layers":1,"d_ff":32}},2)
        self.assertEqual(Model(config).enc_in,2)


if __name__=="__main__":unittest.main()
