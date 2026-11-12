"""Smoke tracking tren CPU, khong phai training; mac dinh chay offline."""

import argparse
import json
import os
import random

from core.tracking import DEFAULT_ENTITY, DEFAULT_PROJECT, init_run
import wandb


def main():
    """Tao run theo config/env, ghi ba metric mau va dong run khi ket thuc."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="offline")
    parser.add_argument("--member", required=True, help="Your own name, used for acceptance evidence")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT))
    args = parser.parse_args()
    if not args.member.strip():
        parser.error("--member must be your non-empty name")
    config = dict(mode=args.mode, seed=args.seed, method="smoke-test", member=args.member,
                  entity=args.entity, project=args.project,
                  name=f"smoke-{args.member}-seed{args.seed}", tags=["smoke-test"], job_type="smoke-test")
    rng = random.Random(args.seed)
    run = init_run(config)
    try:
        for step in range(3):
            metric = {"smoke/value": rng.random(), "smoke/step": step}
            if step == 0:
                wandb.log(metric, step=step, commit=True)
            else:
                run.log(metric, step=step, commit=True)
        run.summary["smoke/completed"] = True
    except BaseException:
        try:
            run.finish(exit_code=1)
        except Exception:
            pass  # Giu loi logging ban dau neu cleanup cung loi.
        raise
    else:
        run.finish()
    print(json.dumps({"mode": args.mode, "entity": args.entity, "project": args.project,
                      "run_id": run.id, "run_dir": run.dir}, ensure_ascii=False))
    if args.mode == "online" and run.url:
        print("Run URL:", run.url)
    else:
        print("No online run URL: this check does not prove server upload.")


if __name__ == "__main__":
    main()
