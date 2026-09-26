"""Command-line entry point.

    python src/run.py prepare  --split all      # Stage 0: parse, normalize, cache
    python src/run.py folds                     # Stage 0: entity-disjoint folds (train)
    python src/run.py block    --split train    # Plan A candidates (+ recall report on train)
    python src/run.py features --split train    # Plan A pair features
    python src/run.py train-a                   # fit + tune Plan A on dev folds, report sealed fold
    python src/run.py submit-a --version v1     # test inference, submission files, validation

  Stage 2 (learned candidate ranker; downstream commands take --variant b):
    python src/run.py rank-train                # union sample from dev folds, 2 cross-fitted rankers
    python src/run.py rank --split train        # ranked candidates (cand_b) + recall@K report
    python src/run.py features --split train --variant b
    python src/run.py train-a --variant b
    python src/run.py rank --split test && python src/run.py features --split test --variant b
    python src/run.py submit-a --variant b --version v2

Paths come from --data-dir / --work-dir or the BER_DATA_DIR / BER_WORK_DIR variables.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

from config import from_args


def _log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def cmd_prepare(cfg, args):
    from prepare import prepare_split
    for split in (["train", "test"] if args.split == "all" else [args.split]):
        prepare_split(cfg, split, _log)


def cmd_translit(cfg, args):
    from prepare import prepare_skeletons
    for split in (["train", "test"] if args.split == "all" else [args.split]):
        prepare_skeletons(cfg, split, _log)


def cmd_folds(cfg, args):
    from prepare import load
    from splits import make_folds
    folds = make_folds(
        load(cfg, "train", "s1", ["s1", "entity_id", "country"]),
        load(cfg, "train", "gt"),
        load(cfg, "train", "tgt", ["t", "script"]),
        cfg.n_folds,
        cfg.seed,
    )
    folds.write_parquet(cfg.split_dir("train") / "folds.parquet")
    _log("fold sizes: " + json.dumps(dict(folds.group_by("fold").len().sort("fold").iter_rows())))
    check = folds.group_by("fold").agg(pl.col("n_true").mean().alias("mean_true"),
                                       (pl.col("n_true") == 0).mean().alias("singleton_rate")).sort("fold")
    print(check)


def cmd_block(cfg, args):
    from blocking import build_candidates
    build_candidates(cfg, args.split, _log)
    if args.split == "train":
        from report import candidate_report
        candidate_report(cfg, _log)


def cmd_rank_train(cfg, args):
    from ranker import train_ranker
    train_ranker(cfg, _log)


def cmd_rank(cfg, args):
    from ranker import rank_split, ranked_report
    rank_split(cfg, args.split, _log)
    if args.split == "train":
        ranked_report(cfg, _log)


def cmd_features(cfg, args):
    from features import build_features
    build_features(cfg, args.split, _log, args.variant)


def cmd_train_a(cfg, args):
    from model_a import train_and_tune
    train_and_tune(cfg, _log, args.variant)


def cmd_report_a(cfg, args):
    from model_a import decide, load_model, scores_path
    from report import strata_report
    model = load_model(cfg, args.variant)
    scores = pl.read_parquet(scores_path(cfg, args.variant), columns=["s1", "t", "p"])
    pred = decide(scores.filter(pl.col("p") >= min(0.15, model["tau"])), model["tau"], model["rho"], model["excl"])
    res = strata_report(cfg, pred, cfg.sealed_fold, _log)
    (cfg.split_dir("train") / f"report_strata_a_{args.variant}.json").write_text(json.dumps(res, indent=2, default=str))


def cmd_submit_a(cfg, args):
    from model_a import predict_split
    from make_submission import make_submission
    pred, cand, model = predict_split(cfg, "test", _log, args.variant)
    make_submission(cfg, args.version, pred, cand, model, _log)


def cmd_train_b(cfg, args):
    from matcher_b import train_and_tune
    train_and_tune(cfg, _log)


def cmd_submit_b(cfg, args):
    from make_submission import make_submission
    from matcher_b import predict_split
    pred, cand, model = predict_split(cfg, "test", _log)
    make_submission(cfg, args.version, pred, cand, model, _log)


def cmd_stack_features(cfg, args):
    from stack import build_stack_features
    build_stack_features(cfg, args.split, _log, args.level)


def cmd_train_c(cfg, args):
    from stack import train_and_tune
    train_and_tune(cfg, _log, args.level)


def cmd_submit_c(cfg, args):
    from make_submission import make_submission
    from stack import predict_split
    pred, cand, model = predict_split(cfg, "test", _log, args.level)
    make_submission(cfg, args.version, pred, cand, model, _log)


def cmd_package(cfg, args):
    from make_submission import final_package
    final_package(cfg, args.version, args.team, Path(args.doc), _log)


def cmd_score_b(cfg, args):
    from matcher_b import split_scores
    split_scores(cfg, args.split, _log)


def cmd_stress(cfg, args):
    from stress import injection, injection_stack
    if args.variant == "c":          # through both stages (v4+); variant "a" keeps the v3 stage-1 test
        injection_stack(cfg, log=_log, pool_folds=args.pool)
    else:
        injection(cfg, log=_log)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "folds", "block", "features", "train-a", "report-a",
                                        "submit-a", "rank-train", "rank", "train-b", "submit-b", "stress",
                                        "score-b", "stack-features", "train-c", "submit-c", "translit",
                                        "package"])
    ap.add_argument("--split", default="train", choices=["train", "test", "all"])
    ap.add_argument("--variant", default="a", choices=["a", "b", "c"],
                    help="a: Plan A candidates; b: ranked candidates; c: ranked candidates + Stage 3 features")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--level", type=int, default=2, choices=[2, 3],
                    help="stacked stage: 2 = on stage-1 scores (v4-v6); 3 = on stage-2 scores")
    ap.add_argument("--pool", default="sealed", choices=["sealed", "dev"],
                    help="stress: evaluate sealed-fold (report) or dev-fold (tuning) entities")
    ap.add_argument("--team", default="team", help="package: zip name prefix (<team>_submission.zip)")
    ap.add_argument("--doc", default="Documentation_template.md", help="package: filled methodology document")
    ap.add_argument("--data-dir")
    ap.add_argument("--work-dir")
    ap.add_argument("--n-jobs", type=int)
    args = ap.parse_args(argv)
    cfg = from_args(args.data_dir, args.work_dir, n_jobs=args.n_jobs)
    _log(f"config {cfg.fingerprint()} data={cfg.data_dir} work={cfg.work_dir} n_jobs={cfg.n_jobs}")
    handler = {
        "prepare": cmd_prepare, "folds": cmd_folds, "block": cmd_block, "features": cmd_features,
        "train-a": cmd_train_a, "report-a": cmd_report_a, "submit-a": cmd_submit_a,
        "rank-train": cmd_rank_train, "rank": cmd_rank, "train-b": cmd_train_b, "submit-b": cmd_submit_b,
        "stress": cmd_stress, "score-b": cmd_score_b, "stack-features": cmd_stack_features,
        "train-c": cmd_train_c, "submit-c": cmd_submit_c, "translit": cmd_translit, "package": cmd_package,
    }[args.command]
    t0 = time.time()
    handler(cfg, args)
    _log(f"{args.command} done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
