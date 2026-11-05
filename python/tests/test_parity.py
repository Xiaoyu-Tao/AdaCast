"""Regression against recorded reference outputs; runs with no original checkout."""
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from adacast_lite_ml.runner import dispatch
from adacast_lite_ml.engine.reflection.controller import derive_policy

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE = json.loads((FIXTURES/"parity-golden.json").read_text())


class ReferenceParityTests(unittest.TestCase):
    def test_recorded_original_predictions_metrics_and_gap_grid(self):
        task = {**REFERENCE["task"], "dataset_path":str(FIXTURES/"parity-series.csv"),
                "frequency":"h", "model_families":["statistical", "deep_learning"]}
        for case in REFERENCE["cases"]:
            if case["model"] in {"DLinear","PatchTST"} and not importlib.util.find_spec("torch"):
                continue
            with self.subTest(model=case["model"]), tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary)
                params={**case["hyperparams"],"seed":42,"device":"cpu","threads":2}
                spec={"model":case["model"],"params":params}
                source=None
                for split in ["val","test"]:
                    request={"protocol_version":1,"operation":"experiment","task":task,"spec":spec,
                             "split":split,"run_id":split,"output_dir":str(root/split)}
                    if source:
                        request.update(source_artifact=source["artifact"],source_sha256=source["artifact_sha256"],source_run_id="val")
                    result=dispatch(request)
                    if split=="val":source=result
                    recorded=case["splits"][split]
                    with open(result["predictions"]) as f:points=list(csv.DictReader(f))
                    np.testing.assert_allclose([float(p["predicted"]) for p in points],recorded["predicted"],rtol=0,atol=1e-7)
                    np.testing.assert_array_equal([float(p["actual"]) for p in points],recorded["actual"])
                    for metric,value in recorded["metrics"].items():
                        self.assertAlmostEqual(result["metrics"][metric],value,places=7)
                    self.assertEqual([[str(pd.Timestamp(p["timestamp"])),int(p["window_index"]),int(p["horizon_index"])] for p in points],[[str(pd.Timestamp(row[0])),row[1],row[2]] for row in recorded["coordinates"]])
                    if split=="test":
                        gap=pd.read_csv(root/split/"gap_predictions.csv")
                        self.assertEqual(len(gap),gap.window_index.nunique()*task["forecast_gap"])
                        self.assertEqual(result["artifact_sha256"],source["artifact_sha256"])

    def test_recorded_original_user_prior_scale_and_global_grid(self):
        expected=json.loads((FIXTURES/"parity-prior-golden.json").read_text())
        task={**REFERENCE["task"],"dataset_path":str(FIXTURES/"parity-series.csv"),"frequency":"h","model_families":["statistical"]}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=None
            for split in ["val","test"]:
                request={"protocol_version":1,"operation":"experiment","task":task,"spec":{"model":"UserPrior","params":{}},"split":split,"run_id":split,"output_dir":str(root/split)}
                if source:request.update(source_artifact=source["artifact"],source_sha256=source["artifact_sha256"],source_run_id="val")
                else:request["prior_path"]=str(FIXTURES/"parity-prior.csv")
                result=dispatch(request)
                if split=="val":source=result
                with open(result["predictions"]) as f:points=list(csv.DictReader(f))
                np.testing.assert_array_equal([float(p["predicted"]) for p in points],expected[split]["predicted"])
                np.testing.assert_array_equal([float(p["actual"]) for p in points],expected[split]["actual"])
                self.assertEqual(sorted({int(p["origin"]) for p in points}),expected[split]["origins"])
                self.assertEqual(result["spec"]["prediction_grid_target_anchor_index"],expected[split]["anchor"])

    def test_recorded_original_calibration_policy(self):
        expected=json.loads((FIXTURES/"parity-policy.json").read_text())
        task={**REFERENCE["task"],"dataset_path":str(FIXTURES/"parity-series.csv"),"reflection_enabled":True}
        actual=derive_policy(task,FIXTURES/"task.json")
        actual.pop("frozen_at",None);expected.pop("frozen_at",None)
        self.assertEqual(actual,expected)


if __name__=="__main__":unittest.main()
