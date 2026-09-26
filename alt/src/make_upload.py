"""Build a leaderboard file from v10-style test scores: optional ownership layer + shift, then
exact expected-F0.5 selection; writes matching_results.tsv + candidate_pairs.tsv and runs the
official validator with --check-ids."""
import argparse, json, os, subprocess, sys
import numpy as np, polars as pl, lightgbm as lgb
ap = argparse.ArgumentParser()
ap.add_argument("--work", default="/home/user/work_test")
ap.add_argument("--out", required=True)
ap.add_argument("--shift", type=float, default=1.0)
ap.add_argument("--owner", action="store_true")
a = ap.parse_args()
from v10env import WORK as V10W, DATA, ROOT
from matcher_b import apply_policy
from data_io import write_id_lists, MATCH_HEADER, CAND_HEADER
import ownership as own

W = f"{a.work}/test"
sc = pl.read_parquet(f"{W}/scores_c.parquet", columns=["s1", "t", "p"])
s1 = pl.read_parquet(f"{W}/s1.parquet", columns=["s1", "entity_id"]).sort("s1")
tg = pl.read_parquet(f"{W}/tgt.parquet", columns=["t", "entity_id", "addr_blank"]).sort("t")
meta = json.load(open(f"{a.work}/models/matcher_c/meta.json"))
pol = {**meta["policy"], "shift": a.shift}
pairs = sc
if a.owner:
    b = {k: lgb.Booster(model_file=f"{V10W}/owner_{k}.txt") for k in "AB"}
    x = own.features(sc.join(tg.select("t", t_blank=pl.col("addr_blank").cast(pl.Float32)), on="t"))
    pairs = x.select("s1", "t", p=pl.Series(own.predict(b, x)))
    pol["name"] = "dta"
pred = apply_policy(pairs.filter(pl.col("p") > 1e-4), pol)
os.makedirs(a.out, exist_ok=True)
ids = lambda df: df.select(s1_id=s1["entity_id"].gather(df["s1"]), t_id=tg["entity_id"].gather(df["t"]))
write_id_lists(__import__("pathlib").Path(f"{a.out}/matching_results.tsv"), MATCH_HEADER, s1["entity_id"], ids(pred))
write_id_lists(__import__("pathlib").Path(f"{a.out}/candidate_pairs.tsv"), CAND_HEADER, s1["entity_id"], ids(sc.select("s1", "t")))
print(f"policy {pol} owner={a.owner}: {pred.height:,} matches for {s1.height:,} S1", flush=True)
r = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", f"{a.out}/matching_results.tsv",
                    "--candidate", f"{a.out}/candidate_pairs.tsv", "--test-dir", "dataset/test", "--check-ids"],
                   cwd=os.path.join(ROOT, "student_resource"), capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-500:])
