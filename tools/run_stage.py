"""Launch experiment stages with their required working directories."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONTROLLED = ROOT / "experiments/controlled"
STAGES = {
    "preliminary": (ROOT / "experiments/preliminary", "run.py", []),
    "stability": (ROOT / "experiments/stability/scripts", "run.py", []),
    "download": (CONTROLLED / "scripts/data_processing", "download_data.py", []),
    "prepare": (CONTROLLED / "scripts/data_processing", "add_patch.py", []),
    "concept": (CONTROLLED / "scripts/data_processing", "generate_patch_set.py", []),
    "testsets": (CONTROLLED / "scripts/data_processing", "export_testset.py", []),
    "train-clean": (CONTROLLED, "scripts/training/train_model.py", ["--cfg", "configs/train_clean.yaml"]),
    "train-spurious": (CONTROLLED, "scripts/training/train_model.py", ["--cfg", "configs/train_spurious.yaml"]),
    "behavior": (CONTROLLED / "scripts/evaluation", "behavior_test.py", []),
    "tcav": (CONTROLLED / "scripts/tcav", "run_tcav.py", []),
    "summarize": (CONTROLLED / "scripts/tcav", "summarize_tcacv_results.py", []),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-dir", type=Path,
                        help="Separate directory for controlled inputs and outputs")
    args = parser.parse_args()
    source, script, arguments = STAGES[args.stage]
    cwd = source
    if source.is_relative_to(CONTROLLED):
        if args.run_dir is None and not args.dry_run:
            parser.error("controlled stages require --run-dir to protect archived evidence")
        if args.run_dir is not None:
            run_dir = args.run_dir.resolve()
            if run_dir.is_relative_to(ROOT) and not run_dir.is_relative_to(ROOT / "out"):
                parser.error("within the repository, use an --run-dir under out/")
            cwd = run_dir / source.relative_to(CONTROLLED)
            if arguments:
                # Prefer run-specific training configs so seeds and outputs can
                # be recorded with the experiment.
                config = run_dir / arguments[1]
                if not config.is_file():
                    config = CONTROLLED / arguments[1]
                arguments = [arguments[0], str(config)]
    elif args.run_dir is not None:
        parser.error("--run-dir is only used by controlled stages")
    command = [sys.executable, str(source / script), *arguments]
    print(f"Working directory: {cwd}", flush=True)
    print(f"Command: {subprocess.list2cmdline(command)}", flush=True)
    if not args.dry_run:
        cwd.mkdir(parents=True, exist_ok=True)
        return subprocess.call(command, cwd=cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
