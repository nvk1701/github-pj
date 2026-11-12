"""Test config va lifecycle bang SDK mock, khong can network."""

from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
import os
import sys
import unittest
from unittest.mock import Mock, patch

from core import tracking


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.sdk = Mock()
        self.sdk.run = None
        self.mock = patch.object(tracking, "wandb", self.sdk)
        self.mock.start()
        self.addCleanup(self.mock.stop)
        self.addCleanup(self.env.stop)

    def test_valid_config_preserved_and_run_returned(self):
        config = {"seed": 42, "method": "linear", "optimizer": {"lr": .1}, "tags": ["linear", "lab1"]}
        before = deepcopy(config)
        self.assertIs(tracking.init_run(config), self.sdk.init.return_value)
        args = self.sdk.init.call_args.kwargs
        self.assertEqual(args["name"], "linear-seed42")
        self.assertEqual(args["tags"], ["lab1", "linear"])
        self.assertEqual(args["config"]["optimizer"], {"lr": .1})
        args["config"]["optimizer"]["lr"] = 10
        self.assertEqual(config, before)

    def test_environment_and_config_precedence(self):
        os.environ.update(WANDB_PROJECT="env-project", WANDB_ENTITY="env-team", WANDB_MODE="offline", WANDB_NAME="env-name")
        tracking.init_run({})
        args = self.sdk.init.call_args.kwargs
        self.assertEqual((args["project"], args["entity"], args["mode"], args["name"]),
                         ("env-project", "env-team", "offline", "env-name"))
        tracking.init_run(dict(project="config-project", entity="config-team", mode="disabled", name="custom", group="g", job_type="j"))
        args = self.sdk.init.call_args.kwargs
        self.assertEqual((args["project"], args["entity"], args["mode"], args["name"], args["group"], args["job_type"]),
                         ("config-project", "config-team", "disabled", "custom", "g", "j"))

    def test_defaults(self):
        tracking.init_run({})
        args = self.sdk.init.call_args.kwargs
        self.assertEqual(args["project"], "ml-distribution-shift")
        self.assertEqual(args["entity"], "gr_ml_29_37_37")
        self.assertEqual(args["reinit"], "return_previous")
        self.assertEqual(args["resume"], "never")
        self.assertEqual(args["tags"], ["lab1"])
        self.assertEqual(args["mode"], "online")

    def test_invalid_config_fails_before_sdk(self):
        invalid = [None, [], "config", {"project": ""}, {"project": None}, {"entity": 3},
                   {"mode": "typo"}, {"mode": None}, {"seed": True}, {"seed": -1},
                   {"seed": 1.5}, {"tags": "linear"}, {"tags": [""]}, {"method": ""},
                   {"name": " "}, {"group": 123}, {"job_type": ""}, {"entity": None},
                   {"entity": " team "}, {"entity": "team/project"}, {"project": "a/b"},
                   {"project": " project"}, {"project": "a\\b"}]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises((TypeError, ValueError)):
                tracking.init_run(config)
        self.sdk.init.assert_not_called()

    def test_credentials_rejected_without_leaking_value(self):
        for config in ({"WANDB_API_KEY": "test-only"}, {"nested": [{"token": "test-only"}]}):
            with self.assertRaises(ValueError) as error:
                tracking.init_run(config)
            self.assertNotIn("test-only", str(error.exception))
        self.sdk.init.assert_not_called()

    def test_sdk_error_propagates(self):
        self.sdk.init.side_effect = RuntimeError("authentication failed")
        with self.assertRaisesRegex(RuntimeError, "authentication failed"):
            tracking.init_run({})

    def test_finish_then_initialize_again(self):
        first, second = Mock(), Mock()
        def activate(configured_run):
            self.sdk.run = configured_run
            configured_run.finish.side_effect = lambda: setattr(self.sdk, "run", None)
            return configured_run
        self.sdk.init.side_effect = lambda **kwargs: activate(first if self.sdk.init.call_count == 1 else second)
        self.assertIs(tracking.init_run({"method": "linear", "tags": ("linear",)}), first)
        with self.assertRaisesRegex(RuntimeError, "already active"):
            tracking.init_run({"method": "convex"})
        self.assertEqual(self.sdk.init.call_count, 1)
        first.finish()
        self.assertIs(tracking.init_run({"method": "convex", "tags": ["convex"]}), second)
        second.finish()
        self.assertEqual(self.sdk.init.call_count, 2)
        self.assertTrue(all(call.kwargs["reinit"] == "return_previous" for call in self.sdk.init.call_args_list))

    def test_experiment_name_and_explicit_override(self):
        tracking.init_run({"experiment": "linear", "method": "optimal", "seed": 0})
        self.assertEqual(self.sdk.init.call_args.kwargs["name"], "linear-optimal-seed0")
        tracking.init_run({"experiment": "convex", "name": "chosen", "group": "figure5", "job_type": "train"})
        args = self.sdk.init.call_args.kwargs
        self.assertEqual((args["name"], args["group"], args["job_type"]), ("chosen", "figure5", "train"))
        self.assertEqual(args["config"]["group"], "figure5")

    def test_explicit_lab2_deduplication_without_mutation(self):
        config = {"lab": "lab2", "tags": ["lab2", "linear", "linear"]}
        before = deepcopy(config)
        tracking.init_run(config)
        self.assertEqual(self.sdk.init.call_args.kwargs["tags"], ["lab2", "linear"])
        self.assertEqual(config, before)

    def test_conflicting_lab_tags_fail(self):
        for config in ({"tags": ["lab2"]}, {"lab": "lab2", "tags": ["lab1"]},
                       {"tags": [" lab1 "]}, {"lab": "lab3"}, {"lab": None}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                tracking.init_run(config)
        self.sdk.init.assert_not_called()

    def test_reserved_sdk_controls_are_rejected(self):
        for option in tracking._UNSUPPORTED_CONTROLS:
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "Unsupported"):
                tracking.init_run({option: "test-option"})
        self.sdk.init.assert_not_called()

    def test_common_secret_aliases_fail_without_values(self):
        for key in ("access_token", "Client-Secret", "APIKey", "Authorization", "refresh_token"):
            with self.subTest(key=key), self.assertRaises(ValueError) as error:
                tracking.init_run({"nested": [{key: "sentinel-do-not-log"}]})
            self.assertNotIn("sentinel-do-not-log", str(error.exception))
        self.sdk.init.assert_not_called()

    def test_finite_json_metadata_only(self):
        for config in ({"value": object()}, {"value": {1, 2}}, {"value": float("nan")},
                       {"value": float("inf")}, {1: "value"}, {"value": b"bytes"}):
            with self.subTest(config_type=type(config)), self.assertRaises((ValueError, TypeError)):
                tracking.init_run(config)
        self.sdk.init.assert_not_called()

    def test_metadata_size_depth_and_cycles_are_bounded(self):
        nested = {}
        for _ in range(12):
            nested = {"nested": nested}
        cycle = []
        cycle.append(cycle)
        for config in ({"value": "x" * 65537}, nested, {"cycle": cycle}, {"nodes": [0] * 10001}):
            with self.assertRaises(ValueError):
                tracking.init_run(config)
        self.sdk.init.assert_not_called()

    def test_invalid_environment_cannot_silently_select_destination(self):
        os.environ["WANDB_ENTITY"] = " "
        with self.assertRaises(ValueError):
            tracking.init_run({})
        self.sdk.init.assert_not_called()

    def test_environment_group_and_job_type_are_recorded(self):
        os.environ.update(WANDB_GROUP="group-from-env", WANDB_JOB_TYPE="train")
        tracking.init_run({})
        self.assertEqual(self.sdk.init.call_args.kwargs["config"]["group"], "group-from-env")
        self.assertEqual(self.sdk.init.call_args.kwargs["config"]["job_type"], "train")

    def test_failed_init_has_no_helper_state_or_retry(self):
        self.sdk.init.side_effect = RuntimeError("sample SDK failure")
        with self.assertRaisesRegex(RuntimeError, "sample SDK failure"):
            tracking.init_run({"mode": "offline"})
        self.sdk.init.assert_called_once()
        self.assertIsNone(self.sdk.run)
        self.sdk.init.side_effect = None
        self.assertIs(tracking.init_run({"mode": "offline"}), self.sdk.init.return_value)

    def test_environment_is_not_copied_into_config(self):
        os.environ["WANDB_API_KEY"] = "sentinel-test-env-only"
        tracking.init_run({"mode": "disabled"})
        self.assertNotIn("WANDB_API_KEY", self.sdk.init.call_args.kwargs["config"])
        self.assertNotIn("sentinel-test-env-only", repr(self.sdk.init.call_args))

    def test_seed_recording_does_not_seed_global_rng(self):
        import random
        state = random.getstate()
        tracking.init_run({"seed": 0})
        self.assertEqual(random.getstate(), state)

    def test_active_external_sdk_run_is_rejected(self):
        self.sdk.run = Mock()
        with self.assertRaisesRegex(RuntimeError, "run.finish"):
            tracking.init_run({})
        self.sdk.init.assert_not_called()
        self.sdk.run.finish.assert_not_called()

    def test_final_config_including_environment_is_size_bounded(self):
        os.environ["WANDB_NAME"] = "x" * 30000  # Khong vuot gioi han do dai environment tren Windows.
        with self.assertRaises(ValueError):
            tracking.init_run({"metadata": "x" * 40000})
        self.sdk.init.assert_not_called()


class SmokeTests(unittest.TestCase):
    def test_smoke_defaults_and_success_cleanup(self):
        from scripts import smoke_tracking
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, "argv", ["smoke", "--member", "Khoi"]), \
                patch.object(smoke_tracking, "init_run") as init, \
                patch.object(smoke_tracking.wandb, "log") as global_log, redirect_stdout(StringIO()):
            init.return_value.id = "mock-id"
            init.return_value.dir = "mock-dir"
            smoke_tracking.main()
        config = init.call_args.args[0]
        self.assertEqual((config["entity"], config["project"], config["mode"]),
                         ("gr_ml_29_37_37", "ml-distribution-shift", "offline"))
        self.assertEqual(global_log.call_count, 1)
        self.assertEqual(init.return_value.log.call_count, 2)
        init.return_value.finish.assert_called_once_with()

    def test_log_failure_closes_run_and_propagates(self):
        from scripts import smoke_tracking
        with patch.object(sys, "argv", ["smoke", "--member", "Khoi"]), \
                patch.object(smoke_tracking, "init_run") as init, \
                patch.object(smoke_tracking.wandb, "log"):
            init.return_value.log.side_effect = RuntimeError("sample log failure")
            with self.assertRaisesRegex(RuntimeError, "sample log failure"):
                smoke_tracking.main()
        init.return_value.finish.assert_called_once_with(exit_code=1)

    def test_empty_member_rejected_before_init(self):
        from scripts import smoke_tracking
        with patch.object(sys, "argv", ["smoke", "--member", " "]), \
                patch.object(smoke_tracking, "init_run") as init, redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as error:
                smoke_tracking.main()
        self.assertEqual(error.exception.code, 2)
        init.assert_not_called()

    def test_global_log_failure_also_closes_run(self):
        from scripts import smoke_tracking
        with patch.object(sys, "argv", ["smoke", "--member", "Khoi"]), \
                patch.object(smoke_tracking, "init_run") as init, \
                patch.object(smoke_tracking.wandb, "log", side_effect=RuntimeError("global failed")):
            with self.assertRaisesRegex(RuntimeError, "global failed"):
                smoke_tracking.main()
        init.return_value.finish.assert_called_once_with(exit_code=1)

    def test_cleanup_failure_preserves_original_logging_error(self):
        from scripts import smoke_tracking
        with patch.object(sys, "argv", ["smoke", "--member", "Khoi"]), \
                patch.object(smoke_tracking, "init_run") as init, \
                patch.object(smoke_tracking.wandb, "log", side_effect=RuntimeError("original failure")):
            init.return_value.finish.side_effect = OSError("teardown failure")
            with self.assertRaisesRegex(RuntimeError, "original failure"):
                smoke_tracking.main()


class NotebookZipTests(unittest.TestCase):
    def setUp(self):
        import ast
        import json
        from pathlib import Path
        notebook = Path(__file__).resolve().parents[1] / "notebooks/T2_environment_wandb.ipynb"
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        tree = ast.parse("".join(cells[1]["source"]))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "extract_source_zip")
        namespace = {}
        # Chi lay ham giai nen de test, khong chay code Colab hay login.
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<notebook-zip>", "exec"), namespace)
        self.extract = namespace["extract_source_zip"]

    def archive(self, entries):
        """Tao ZIP nho trong bo nho de test duong dan, khong can file source that."""
        import zipfile
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, value in entries:
                if isinstance(name, str):
                    entry = zipfile.ZipInfo()
                    entry.filename = name
                    entry.orig_filename = name
                    name = entry
                archive.writestr(name, value)
        buffer.seek(0)
        archive = zipfile.ZipFile(buffer)
        # Giu ten raw nhu khi doc tren Linux, vi zipfile Windows tu doi backslash.
        for entry in archive.infolist():
            entry.filename = entry.orig_filename
        return archive

    def test_windows_and_posix_source_paths(self):
        import tempfile
        from pathlib import Path
        for separator in ("/", "\\"):
            with self.subTest(separator=separator), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with self.archive([("source" + separator, ""),
                                   ("source" + separator + "requirements.txt", "wandb==0.31.0"),
                                   (separator.join(["source", "core", "tracking.py"]), "pass")]) as archive:
                    self.extract(archive, root)
                self.assertEqual((root / "source/core/tracking.py").read_text(), "pass")
                self.assertTrue((root / "source/requirements.txt").is_file())

    def test_unsafe_paths_rejected_before_writing(self):
        import tempfile
        from pathlib import Path
        for name in ("../outside", "..\\outside", "/absolute", "C:\\outside", "\\\\server\\file", "safe/../outside"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with self.archive([("safe.txt", "safe"), (name, "bad")]) as archive:
                    with self.assertRaises(ValueError):
                        self.extract(archive, root)
                self.assertEqual(list(root.iterdir()), [])

    def test_duplicate_normalized_paths_rejected(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temporary:
            with self.archive([("a\\b", "first"), ("a/b", "second")]) as archive:
                with self.assertRaises(ValueError):
                    self.extract(archive, Path(temporary))
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_symlink_rejected(self):
        import stat
        import tempfile
        import zipfile
        from pathlib import Path
        entry = zipfile.ZipInfo("link")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        with tempfile.TemporaryDirectory() as temporary:
            with self.archive([(entry, "../outside")]) as archive:
                with self.assertRaises(ValueError):
                    self.extract(archive, Path(temporary))

    def test_file_directory_collision_rejected(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temporary:
            with self.archive([("a", "file"), ("a/b", "child")]) as archive:
                with self.assertRaises(ValueError):
                    self.extract(archive, Path(temporary))
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
