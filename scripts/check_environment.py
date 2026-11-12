"""Kiem tra import va phien ban package theo requirements."""

import importlib
from importlib.metadata import metadata, version
from pathlib import Path
import platform
import argparse
import sys


def main():
    """Kiem tra Python, venv, pins va tuong thich CPU; GPU chi test khi co CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-isolated", action="store_true", help="Require a venv without system packages")
    parser.add_argument("--gpu", action="store_true", help="Try one CUDA tensor only if CUDA is available")
    args = parser.parse_args()
    print("Python:", platform.python_version(), platform.platform())
    print("Interpreter:", sys.executable)
    print("Architecture:", platform.machine())
    modules = {"scikit-learn": "sklearn", "pyyaml": "yaml"}
    errors = []
    if sys.version_info[:2] != (3, 12):
        errors.append("official target is CPython 3.12")
    if args.require_isolated:
        cfg = Path(sys.prefix) / "pyvenv.cfg"
        if (sys.prefix == sys.base_prefix or not cfg.is_file()
                or "include-system-site-packages = false" not in cfg.read_text().lower()):
            errors.append("isolated venv required (system-site-packages must be false)")
    for line in (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        package, expected = line.split("==")
        try:
            importlib.import_module(modules.get(package, package))
            actual = version(package)
            from packaging.specifiers import SpecifierSet
            requires_python = metadata(package).get("Requires-Python", "")
            if requires_python and platform.python_version() not in SpecifierSet(requires_python):
                errors.append(package + " Python constraint")
            # Chap nhan build CPU/CUDA cung release, vi du +cpu hoac +cu128.
            matches = actual.split("+")[0] == expected
            print(f"{package}: {actual} (expected {expected}) {'PASS' if matches else 'FAIL'}")
            if not matches:
                errors.append(package)
        except Exception as exc:
            print(f"{package}: FAIL ({type(exc).__name__}: {exc})")
            errors.append(package)
    if errors:
        raise SystemExit("Environment check failed: " + ", ".join(errors))

    import numpy as np
    import scipy.linalg
    import torch
    import pandas as pd
    import pyarrow as pa
    import yaml
    from sklearn.linear_model import LinearRegression
    from datasets import Dataset
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    x = np.array([[1.], [2.], [3.]])
    y = 2 * x.ravel()
    assert np.allclose(LinearRegression().fit(x, y).predict(x), y)
    assert np.allclose(scipy.linalg.solve(np.eye(2), np.ones(2)), np.ones(2))
    assert np.array_equal(torch.from_numpy(x).numpy(), x)
    assert torch.tensor([1., 2.]).square().sum().item() == 5.0
    table = pa.Table.from_pandas(pd.DataFrame({"x": x.ravel(), "y": y}))
    assert Dataset(table).to_pandas().shape == (3, 2)
    assert yaml.safe_load(yaml.safe_dump({"seed": 42})) == {"seed": 42}
    figure = Figure()
    FigureCanvasAgg(figure)
    figure.subplots().plot(x, y)
    figure.canvas.draw()
    print("CPU numerical/NumPy-Torch/pandas-Arrow-datasets/Agg/YAML checks: PASS")
    print("Torch build:", torch.__version__, "compiled CUDA:", torch.version.cuda,
          "CUDA available:", torch.cuda.is_available())
    if args.gpu:
        if torch.cuda.is_available():
            try:
                value = torch.tensor([1., 2.], device="cuda").square().sum().item()
                torch.cuda.synchronize()
                assert value == 5.0
                print("GPU tensor smoke: PASS", torch.cuda.get_device_name(0))
            except Exception as exc:
                raise SystemExit("GPU tensor smoke: FAIL (" + type(exc).__name__ + ")") from None
        else:
            print("GPU tensor smoke: SKIPPED (CUDA unavailable in this Torch build/runtime)")


if __name__ == "__main__":
    main()
