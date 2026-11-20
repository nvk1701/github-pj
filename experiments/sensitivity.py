"""Xuat hinh tu CSV frozen vao thu muc moi, khong huan luyen hay ghi de."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_SOURCE = "7e0499c450a3e503ab09d4c6c3d272e6576b4432"
PUBLICATION_COMMIT = "4d6bce54e68e559d77cb64bd5dab5eeb1cf7fac6"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_sources(source):
    """Kiem tra hash truoc khi doc CSV; khong dung du lieu khac publication."""
    manifest_path = source / "publication_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("scientific_source_head") != FROZEN_SOURCE
            or manifest.get("checkpoint") != "CP05.6"):
        raise ValueError("Expected the CP05.6 publication with frozen scientific source")
    entries = {row["path"]: row for row in manifest["files"]}
    tables, hashes = {}, {}
    for name in ("cp05_condition_summary.csv", "cp05_per_seed.csv",
                 "d512_diagnostic_summary.csv"):
        path = source / "data" / name
        # Git luu LF; checkout Windows co the dung CRLF cho van ban.
        canonical = path.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha256(canonical).hexdigest()
        if digest != entries["data/" + name]["sha256"]:
            raise ValueError("Published CSV checksum mismatch: " + name)
        hashes["data/" + name] = digest
        with path.open(encoding="utf-8", newline="") as stream:
            tables[name] = list(csv.DictReader(stream))
    return tables, hashes, _sha256(manifest_path)


def _plot_conditions(rows, conditions, xlabel, title, path):
    """Bieu dien mean va sample SD, khong thay doi so lieu trong CSV."""
    selected = {row["condition"]: row for row in rows}
    values = [float(selected[key]["mean_B"]) for key in conditions]
    errors = [float(selected[key]["sample_SD_B"]) for key in conditions]
    labels = [str(float(selected[key][xlabel])) for key in conditions]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:
        ax.bar(np.arange(len(conditions)), values, yerr=errors, capsize=5)
        ax.set_xticks(np.arange(len(conditions)), labels)
        ax.set(xlabel="ell (sample-and-hold)" if xlabel == "ell" else "Label-noise standard deviation",
               ylabel="Final cumulative population B (mean +/- sample SD)", title=title)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def generate_sensitivity_figures_and_tables(output_dir=None):
    """Doc CSV CP05, xuat hinh/bang/provenance vao output root chua ton tai.

    Tu choi root da co, root trong publication hoac root bao trum publication.
    Khong tao run W&B, khong sua source CSV/PNG va khong huan luyen lai.
    Tra ve provenance mapping sau khi xuat thanh cong; loi duoc truyen ra caller.
    """
    source = (PROJECT_ROOT / "docs/results/cp04-cp05").resolve()
    output = Path(output_dir or PROJECT_ROOT / "outputs/sensitivity-derived").resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output root must be separate from the frozen publication")
    if output.exists():
        raise FileExistsError("Output root must be new: " + str(output))
    tables, source_hashes, manifest_hash = _read_sources(source)
    diagnostic = [row for row in tables["d512_diagnostic_summary.csv"]
                  if int(row["dimension"]) == 512]
    if len(diagnostic) != 3 or {row["method"] for row in diagnostic} != {
            "convex", "constant", "inverse_sqrt"}:
        raise ValueError("Expected exactly three d512 method groups")
    output.mkdir(parents=True, exist_ok=False)
    for name in tables:
        shutil.copyfile(source / "data" / name, output / name)
    conditions = tables["cp05_condition_summary.csv"]
    _plot_conditions(conditions, ("ell1", "baseline", "ell16"), "ell",
                     "CP05 Convex-only OFAT: d128, three seeds", output / "drift_sensitivity.png")
    _plot_conditions(conditions, ("noise0", "baseline", "noise1"), "label_noise_std",
                     "CP05 noise conditions: distribution and comparator both vary",
                     output / "noise_sensitivity.png")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:
        ax.bar([row["method"] for row in diagnostic],
               [float(row["mean_B"]) for row in diagnostic],
               yerr=[float(row["sample_SD_B"]) for row in diagnostic], capsize=5)
        ax.set(title="d512: three method groups, three seeds each",
               ylabel="Final cumulative population B (mean +/- sample SD)")
        fig.tight_layout()
        fig.savefig(output / "d512_method_summary.png", dpi=160)
    finally:
        plt.close(fig)
    head = subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True)
    state = subprocess.run(["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
                           capture_output=True, text=True)
    provenance = {
        "scientific_source_sha": FROZEN_SOURCE,
        "publication_commit": PUBLICATION_COMMIT,
        "publication_manifest_file_sha256": manifest_hash,
        "plotting_tool_head": head.stdout.strip() if head.returncode == 0 else None,
        "plotting_worktree_dirty": bool(state.stdout.strip()) if state.returncode == 0 else None,
        "plotting_script_file_sha256": _sha256(Path(__file__)),
        "source_csv_canonical_lf_sha256": source_hashes,
        "d512_groups": [{"dimension": int(row["dimension"]), "method": row["method"]}
                        for row in diagnostic],
        "scope": "New derived presentation only; original publication remains unchanged",
        "d512_figure_note": "Final method summary; does not replace frozen loss-dynamics curves",
        "files_sha256": {path.name: _sha256(path) for path in sorted(output.iterdir())},
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def main():
    """Xu ly CLI truoc khi ghi file; --help khong tao output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        help="New directory outside the frozen publication; existing roots fail")
    args = parser.parse_args()
    print(json.dumps(generate_sensitivity_figures_and_tables(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
