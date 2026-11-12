"""Kiem tra SDK that trong process rieng voi artifacts offline tam thoi."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class OfflineSDKTests(unittest.TestCase):
    def test_global_logging_lifecycle_and_lab2(self):
        root = Path(__file__).resolve().parents[1]
        # Bo settings W&B ke thua; process con chi chay offline.
        env = {key: value for key, value in os.environ.items() if not key.startswith("WANDB_")}
        with tempfile.TemporaryDirectory(prefix="t2-offline-") as temporary:
            work = Path(temporary)
            env.update(WANDB_MODE="offline", WANDB_DIR=str(work),
                       WANDB_CACHE_DIR=str(work / "cache"), WANDB_CONFIG_DIR=str(work / "config"),
                       WANDB_DATA_DIR=str(work / "data"), WANDB_API_KEY="",
                       WANDB_BASE_URL="https://api.forge.coreweave.com",
                       WANDB_DISABLE_GIT="true", WANDB_CONSOLE="off", WANDB_SILENT="true",
                       PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
            script = r'''
import json
import time
import os
from pathlib import Path
import wandb
from core.tracking import init_run
from wandb.sdk.lib.wbauth import check_api_key

# Chi kiem tra format bang chuoi gia, khong login hay goi HTTP.
assert wandb.Settings().base_url == "https://api.forge.coreweave.com"
assert check_api_key("A" * 86) is None
assert check_api_key("A" * 40) is None
assert check_api_key("A" * 39) is not None

def assert_summary(run, expected):
    # Cho backend cap nhat summary sau khi nhan metric.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if all(run.summary.get(key) == value for key, value in expected.items()):
            return
        time.sleep(0.05)
    raise AssertionError("logged metrics did not reach the public run summary")

assert wandb.run is None
run = init_run(dict(mode="offline", experiment="linear", method="optimal", seed=0,
                    group="linear-figure5", job_type="test", tags=["linear", "linear"]))
first_id = run.id
first_dir = run.dir
try:
    assert wandb.run is run
    assert run.settings.base_url == "https://api.forge.coreweave.com"
    assert run.settings.mode == "offline"
    assert run.entity == "gr_ml_29_37_37" and run.project == "ml-distribution-shift"
    assert run.name == "linear-optimal-seed0" and list(run.tags) == ["lab1", "linear"]
    assert run.group == "linear-figure5" and run.config["seed"] == 0
    # Co step thi SDK mac dinh commit=False; can commit truoc khi doc summary.
    wandb.log({"contract/global": 0.5}, step=0, commit=True)
    run.log({"contract/object": 0.01}, step=1, commit=True)
    # Doc summary cong khai, khong phu thuoc format history noi bo.
    assert_summary(run, {"contract/global": 0.5, "contract/object": 0.01})
    try:
        init_run({"mode": "offline"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("second active init was not rejected")
    assert wandb.run is run and wandb.run.id == first_id
finally:
    run.finish()
assert wandb.run is None

second = init_run(dict(mode="offline", lab="lab2", name="explicit-lab2", tags=["lab2"]))
second_id = second.id
second_dir = second.dir
try:
    assert second_id != first_id and wandb.run is second
    assert list(second.tags) == ["lab2"] and "lab1" not in second.tags
    wandb.log({"contract/lab2": 1}, step=0, commit=True)
    assert_summary(second, {"contract/lab2": 1})
finally:
    second.finish()
assert wandb.run is None

for directory in [first_dir, second_dir]:
    parent = Path(directory).parent
    assert "offline-run-" in parent.name
    assert parent.is_dir() and any(p.is_file() for p in parent.rglob("*"))
assert len(list(Path(os.environ["WANDB_DIR"]).glob("wandb/offline-run-*"))) == 2
disabled = init_run({"mode": "disabled"})
try:
    assert wandb.run is disabled
    wandb.log({"disabled/global": 1})
    disabled.log({"disabled/object": 2})
finally:
    disabled.finish()
assert wandb.run is None
# Dong service bang API cong khai truoc khi don artifacts tam.
wandb.teardown()
print(json.dumps({"sdk": wandb.__version__, "mode": "offline", "uploaded": False,
                  "run_ids": [first_id, second_id], "global_and_object_metrics": True,
                  "disabled_logging": True, "synthetic_86_key_format": True,
                  "default_api": wandb.Settings().base_url}))
'''
            # Service co the giu cwd tren Windows sau khi process con ket thuc.
            # Giu cwd tai repo; artifacts van nam trong cac duong dan tam WANDB_*.
            result = subprocess.run([sys.executable, "-B", "-c", script], cwd=root,
                                    env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            evidence = json.loads(result.stdout.strip().splitlines()[-1])
            expected = dict(line.split("==") for line in (root / "requirements.txt").read_text().splitlines()
                            if line.strip() and not line.startswith("#"))["wandb"]
            self.assertEqual(evidence["sdk"], expected)
            self.assertEqual(evidence["mode"], "offline")
            self.assertFalse(evidence["uploaded"])
            self.assertTrue(evidence["global_and_object_metrics"])
            self.assertEqual(len(set(evidence["run_ids"])), 2)
            self.assertTrue(evidence["disabled_logging"])
            self.assertTrue(evidence["synthetic_86_key_format"])
            self.assertEqual(evidence["default_api"], "https://api.forge.coreweave.com")
            print("Offline SDK evidence:", json.dumps(evidence))


if __name__ == "__main__":
    unittest.main()
