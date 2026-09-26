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
ap.add_argument("--country-shift", default="", help="e.g. US=1.0,India=0.45")
ap.add_argument("--keep-rows-from", default="", help="copy rows of these countries from a base file, e.g. France=../../v10/matching_results.tsv")
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
if a.country_shift:
    cs = {k: float(v) for k, v in (kv.split("=") for kv in a.country_shift.split(","))}
    cty = pl.read_parquet(f"{W}/s1.parquet", columns=["s1", "country"])
    r = pl.col("country").replace_strict(cs, default=a.shift, return_dtype=pl.Float64)
    odds = pl.col("p") / (1 - pl.col("p")).clip(1e-12, None)
    pairs = pairs.join(cty, on="s1").with_columns(p=(odds * r) / (1 + odds * r)).drop("country")
    pol["shift"] = 1.0
pred = apply_policy(pairs.filter(pl.col("p") > 1e-4), pol)
os.makedirs(a.out, exist_ok=True)
ids = lambda df: df.select(s1_id=s1["entity_id"].gather(df["s1"]), t_id=tg["entity_id"].gather(df["t"]))
write_id_lists(__import__("pathlib").Path(f"{a.out}/matching_results.tsv"), MATCH_HEADER, s1["entity_id"], ids(pred))
import shutil
if not os.path.exists(f"{a.out}/candidate_pairs.tsv"):
    shutil.copy("/home/user/Amazon_ml/alt/output/v10_owner_s1/candidate_pairs.tsv", f"{a.out}/candidate_pairs.tsv")
print(f"policy {pol} owner={a.owner}: {pred.height:,} matches for {s1.height:,} S1", flush=True)
if a.keep_rows_from:
    c, path = a.keep_rows_from.split("=")
    rd = lambda f: pl.read_csv(f, separator="\t", quote_char=None, infer_schema=False).fill_null("")
    new_m, base = rd(f"{a.out}/matching_results.tsv"), rd(path)
    keep = pl.read_parquet(f"{W}/s1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == c)["entity_id"]
    base_c = base.filter(pl.col("source1_entity_id").is_in(keep))
    m = new_m.join(base_c, on="source1_entity_id", how="left", suffix="_b").with_columns(
        matched_entity_ids=pl.coalesce("matched_entity_ids_b", "matched_entity_ids")).drop("matched_entity_ids_b")
    m.write_csv(f"{a.out}/matching_results.tsv", separator="\t", quote_style="never", line_terminator="\n")
    print(f"copied {base_c.height:,} {c} rows from {path}", flush=True)
r = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", f"{a.out}/matching_results.tsv",
                    "--candidate", f"{a.out}/candidate_pairs.tsv", "--test-dir", "dataset/test", "--check-ids"],
                   cwd=os.path.join(ROOT, "student_resource"), capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-500:])
