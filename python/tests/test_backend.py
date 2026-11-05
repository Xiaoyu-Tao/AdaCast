import csv
import json
import math
import tempfile
import unittest
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from adacast_lite_ml.data import Data, sha_file
from adacast_lite_ml.runner import dispatch


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.file = self.root / "data.csv"
        with self.file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "target", "label"])
            for i in range(360):
                writer.writerow([(datetime(2025,1,1)+timedelta(hours=i)).isoformat(), 10+math.sin(i/4)+i/100, "context"])
        self.task = {"dataset_path": str(self.file), "time_col": "date", "target_col": "target", "frequency": "h",
                     "seq_len": 24, "pred_len": 12, "forecast_gap": 2, "window_stride": 6,
                     "split_mode": "ratio", "train_ratio": .6, "val_ratio": .2, "test_ratio": .2,
                     "model_families": ["Naive", "Linear"]}
    def tearDown(self):
        self.tmp.cleanup()
    def call(self, operation, **kwargs):
        return dispatch({"protocol_version":1,"operation_id":"unit-test","operation":operation,"task":self.task,**kwargs})
    def train(self, model="Linear"):
        return self.call("experiment",spec={"model":model,"params":{}},split="val",run_id="val",output_dir=str(self.root/"val"))
    def test_training_scale_does_not_use_future_targets(self):
        before = Data(self.task)
        with self.file.open() as f:
            rows = list(csv.DictReader(f))
        for row in rows[before.t:]:
            row["target"] = "9999999"
        with self.file.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=before.columns); w.writeheader(); w.writerows(rows)
        after=Data(self.task)
        self.assertEqual(before.mean,after.mean)
        self.assertEqual(before.scale,after.scale)
    def test_timestamp_split_and_forecast_coordinates(self):
        data=Data(self.task)
        task={k:v for k,v in self.task.items() if k not in ("train_ratio","val_ratio","test_ratio")}
        task.update(split_mode="timestamp",train_end_timestamp=data.times[215],validation_end_timestamp=data.times[287])
        self.assertEqual(Data(task).boundaries(),data.boundaries())
        result=self.train("Naive")
        with open(result["predictions"]) as f:
            point=next(csv.DictReader(f))
        self.assertEqual(int(point["origin"]),252)
        self.assertEqual(int(point["row_index"]),254)
    def test_test_inference_reuses_exact_model_and_spec(self):
        trained=self.train()
        before=sha_file(trained["artifact"])
        result=self.call("experiment",spec=trained["spec"],split="test",run_id="test",output_dir=str(self.root/"test"),
                         source_artifact=trained["artifact"],source_sha256=before,source_run_id="val")
        self.assertEqual(result["artifact_sha256"],before)
        self.assertEqual(sha_file(trained["artifact"]),before)
        with self.assertRaisesRegex(ValueError,"exactly match"):
            self.call("experiment",spec={"model":"Linear","params":{"ridge":1}},split="test",run_id="bad",output_dir=str(self.root/"bad"),source_artifact=trained["artifact"],source_sha256=before)
    def test_test_split_cannot_fit(self):
        with self.assertRaisesRegex(ValueError,"only allowed"):
            self.call("experiment",spec={"model":"Naive","params":{}},split="test",run_id="bad",output_dir=str(self.root/"bad"))
    def test_diagnostics_reject_held_out_data_and_blind_windows_omit_actuals(self):
        trained=self.train()
        with self.assertRaisesRegex(ValueError,"validation-front"):
            self.call("diagnostics",predictions=trained["predictions"])
        windows=self.call("blind_windows",predictions=trained["predictions"])
        self.assertNotIn('"actual"',json.dumps(windows))
        self.assertNotIn('"actual_original"',json.dumps(windows))
        self.assertEqual(len(windows[0]["points"]),12)
    def test_prior_requires_identical_non_target_columns(self):
        with self.file.open() as f:
            rows=list(csv.DictReader(f))
        rows[300]["label"]="changed"
        prior=self.root/"prior.csv"
        with prior.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["date","target","label"]);w.writeheader();w.writerows(rows)
        with self.assertRaisesRegex(ValueError,"may only replace"):
            self.call("experiment",spec={"model":"UserPrior","params":{}},split="val",run_id="prior",output_dir=str(self.root/"prior"),prior_path=str(prior))
    def test_portable_prediction_uses_saved_training_scale(self):
        trained=self.train()
        data=Data(self.task)
        result=self.call("predict",model_artifact=trained["artifact"],history=data.values[-24:])
        self.assertEqual(len(result["values"]),14)
        self.assertTrue(all(math.isfinite(v) for v in result["values"]))
    def test_source_hash_tamper_is_detected(self):
        trained=self.train()
        Path(trained["artifact"]).write_text("{}")
        with self.assertRaisesRegex(ValueError,"provenance"):
            self.call("experiment",spec=trained["spec"],split="test",run_id="bad",output_dir=str(self.root/"bad"),source_artifact=trained["artifact"],source_sha256=trained["artifact_sha256"])
    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional neural dependency is not installed")
    def test_dlinear_checkpoint_is_reused_without_fitting(self):
        from unittest.mock import patch
        spec={"model":"DLinear","params":{"epochs":1,"batch_size":16,"kernel":5,"seed":42}}
        trained=self.call("experiment",spec=spec,split="val",run_id="neural-val",output_dir=str(self.root/"neural-val"))
        model=json.loads(Path(trained["artifact"]).read_text())
        self.assertIn("checkpoint.pth",model["files"])
        with patch("adacast_lite_ml.runner.fit", side_effect=AssertionError("test must not fit")):
            tested=self.call("experiment",spec=spec,split="test",run_id="neural-test",output_dir=str(self.root/"neural-test"),source_artifact=trained["artifact"],source_sha256=trained["artifact_sha256"])
        self.assertTrue(math.isfinite(tested["metrics"]["mse"]))
        self.assertEqual(sha_file(trained["artifact"]),trained["artifact_sha256"])


if __name__ == "__main__":
    unittest.main()
