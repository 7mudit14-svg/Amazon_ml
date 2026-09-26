"""France self-training (recipe validated by LOCO: +0.0039 on an unseen country).

Stage 2 is refit on labelled train rows (US+India dev folds, dom=0) plus France test pseudo-labels
(dom=1): positives = v10 stage-2 p >= 0.98 with agreeing house number, negatives = p <= 0.02.
Test labels are never used (there are none); pseudo-labels come only from our own model.
France pairs are rescored with dom=1 and decided with v10's rule (single_owner + exact expected
F0.5, shift 0.6, the LOCO-best shift). Output: France pairs (s1, t) in test indices.
"""
import glob, sys, time
import numpy as np, polars as pl, lightgbm as lgb
from v10env import WORK
from stack import FEATURES_D, PARAMS
from decide import dta_select, prior_shift, single_owner

t0 = time.time(); T = f"{WORK}/train"; W = "/home/user/work_test/test"
F2 = list(FEATURES_D) + ["dom"]
keep_neg = (pl.struct("s1", "t").hash(seed=5) % 100) < 10
tr = pl.concat([pl.read_parquet(p).filter((pl.col("fold") != 4) & ((pl.col("label") == 1) | keep_neg))
                for p in sorted(glob.glob(f"{T}/feat_d/part_*.parquet"))[:35]]).with_columns(dom=pl.lit(0.0, pl.Float32))
va = pl.concat([pl.read_parquet(p).filter(pl.col("fold") == 4) for p in sorted(glob.glob(f"{T}/feat_d/part_*.parquet"))[35:38]]
               ).sample(fraction=0.3, seed=1).with_columns(dom=pl.lit(0.0, pl.Float32))
fr_s1 = pl.read_parquet(f"{W}/s1.parquet", columns=["s1", "country"]).filter(pl.col("country") == "France").select("s1")
fr = pl.concat([pl.read_parquet(p).join(fr_s1, on="s1", how="semi") for p in sorted(glob.glob(f"{W}/feat_d/part_*.parquet"))])
p2 = pl.read_parquet(f"{W}/scores_c.parquet", columns=["s1", "t", "p"]).join(fr_s1, on="s1", how="semi")
fr = fr.join(p2, on=["s1", "t"]).with_columns(dom=pl.lit(1.0, pl.Float32))
print(f"train {tr.height:,} val {va.height:,} France pairs {fr.height:,} ({time.time()-t0:.0f}s)", flush=True)
fr = fr.select(["s1", "t", "p"] + F2)
pos = fr.filter((pl.col("p") >= 0.98) & (pl.col("num_match") == 1)).with_columns(label=pl.lit(1, pl.UInt8))
neg = fr.filter((pl.col("p") <= 0.02) & keep_neg).with_columns(label=pl.lit(0, pl.UInt8))  # same 10% negative sampling as train
ps = pl.concat([pos, neg])
print(f"France pseudo-labels: pos {pos.height:,} neg {neg.height:,}", flush=True)
X = pl.concat([tr.select(F2 + ["label"]), ps.select(F2 + ["label"])])
wt = np.where(X["label"].to_numpy() == 1, 1.0, 10.0) * np.r_[np.ones(tr.height), np.full(ps.height, 2.0)]  # pseudo rows x2 (LOCO best)
d = lgb.Dataset(X.select(F2).to_numpy(), X["label"].to_numpy(), weight=wt, feature_name=F2)
dv = lgb.Dataset(va.select(F2).to_numpy(), va["label"].to_numpy(), weight=np.where(va["label"].to_numpy() == 1, 1.0, 10.0), reference=d)
b = lgb.train(PARAMS, d, num_boost_round=800, valid_sets=[dv], callbacks=[lgb.early_stopping(30, verbose=False)])
b.save_model(f"{WORK}/france_stage2.txt")
q = b.predict(fr.select(F2).to_numpy())
sc = fr.select("s1", "t").with_columns(p=pl.Series(q))
pred = dta_select(single_owner(prior_shift(sc, 0.6)).filter(pl.col("p") > 1e-4), 0.0).select("s1", "t")
pred.write_parquet(f"{WORK}/france_pred_w2.parquet")
old = dta_select(single_owner(prior_shift(p2, 0.6)).filter(pl.col("p") > 1e-4), 0.0).select("s1", "t")
agree = pred.join(old, on=["s1", "t"]).height
print(f"iters {b.best_iteration}; France matches new {pred.height:,} vs v10 {old.height:,}, common {agree:,} ({time.time()-t0:.0f}s)")
