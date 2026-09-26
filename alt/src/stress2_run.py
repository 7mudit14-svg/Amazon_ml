"""Run v10's unchanged test path (or the patched copy) on a stress-v2 split and score it.

Scored S1s: visible fold-4 S1s of the sample. Reference: the same S1s under v10's stored
out-of-fold scores at full train density (the local 0.9761 setting).
"""
import argparse, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--work", default="/home/user/work_s2")
ap.add_argument("--src", default="/home/user/Amazon_ml/alt/v10p/src")
ap.add_argument("--steps", default="rank,features,scoreb,stack")
ap.add_argument("--shift", type=float, default=None)
a = ap.parse_args()
sys.path.insert(0, a.src)
import polars as pl
from config import from_args
from v10env import WORK as V10W, DATA
from score import per_entity

cfg = from_args(DATA, a.work, n_jobs=4)
os.makedirs(f"{a.work}/models", exist_ok=True)
for m in ("ranker", "matcher_b", "matcher_c"):
    if not os.path.exists(f"{a.work}/models/{m}"):
        os.symlink(f"/home/user/work_test/models/{m}", f"{a.work}/models/{m}")
t0 = time.time(); log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
steps = a.steps.split(",")
if "rank" in steps:
    from ranker import rank_split; rank_split(cfg, "test", log)
if "features" in steps:
    from features import build_features; build_features(cfg, "test", log, "c")
if "scoreb" in steps:
    from matcher_b import split_scores
    (cfg.split_dir("test") / "scores_b.parquet").unlink(missing_ok=True); split_scores(cfg, "test", log)
if "stack" in steps:
    from stack import build_stack_features, level_scores
    (cfg.split_dir("test") / "scores_c.parquet").unlink(missing_ok=True)
    build_stack_features(cfg, "test", log, 2); level_scores(cfg, "test", 2, log)

from matcher_b import apply_policy
from decide import dta_select, prior_shift
import lightgbm as lgb
import ownership as own
from v10env import tgt_train
meta = json.load(open(f"/home/user/work_test/models/matcher_c/meta.json"))
gt = pl.read_parquet(f"{V10W}/train/gt.parquet")
folds = pl.read_parquet(f"{V10W}/train/folds.parquet", columns=["s1", "fold"])
ms1 = pl.read_parquet(f"{a.work}/map_s1.parquet"); mt = pl.read_parquet(f"{a.work}/map_t.parquet")
ev = ms1.join(folds, left_on="orig_s1", right_on="s1").filter(pl.col("fold") == 4).select(s1="orig_s1")
sc = pl.read_parquet(f"{a.work}/test/scores_c.parquet").select("s1", "t", "p")
ref_sc = pl.read_parquet(f"{V10W}/train/scores_c.parquet", columns=["s1", "t", "p"])
tb = tgt_train().select("t", a="business_address").with_columns(
    t_blank=(pl.col("a").is_null() | pl.col("a").str.to_lowercase().is_in(["none", "null", "n/a", ""])).cast(pl.Float32)).select("t", "t_blank")
boost = {k: lgb.Booster(model_file=f"{V10W}/owner_{k}.txt") for k in "AB"}

def owner_scores(scores, tmap=None):
    x = scores
    x = x.join(tmap, on="t").join(tb, left_on="orig_t", right_on="t").drop("orig_t") if tmap is not None else x.join(tb, on="t")
    x = own.features(x.select("s1", "t", "p", "t_blank"))
    return x.select("s1", "t", p=pl.Series(own.predict(boost, x)))

def score(pairs, remap, shift, owner_renorm):
    pol = {**meta["policy"], "shift": shift}
    if not owner_renorm: pol["name"] = "dta"
    pred = apply_policy(pairs.filter(pl.col("p") > 1e-4), pol)
    if remap: pred = pred.join(ms1, on="s1").join(mt, on="t").select(s1="orig_s1", t="orig_t")
    return per_entity(ev, pred, gt)["f"].mean()

own_s2 = owner_scores(sc, mt)
own_ref = owner_scores(ref_sc)
for shift in (1.0, 0.8, 0.6, 0.45, 0.3):
    r1, s1_ = score(ref_sc, False, shift, True), score(sc, True, shift, True)
    r2, s2_ = score(own_ref, False, shift, False), score(own_s2, True, shift, False)
    print(f"RESULT shift {shift}: v10 local {r1:.5f} test-density {s1_:.5f} (drop {r1-s1_:+.5f}) | "
          f"owner local {r2:.5f} test-density {s2_:.5f} (drop {r2-s2_:+.5f})", flush=True)
