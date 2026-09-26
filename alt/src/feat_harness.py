"""Quick A/B for new stage-2 features: refit stage 2 on dev folds of feat_d parts, score fold 4.

usage: python feat_harness.py N_PARTS   (new features come from newfeats.add(df) if present)
Base = v10 FEATURES_D; B = FEATURES_D + newfeats.NEW. Same rows, same params, same decision rule
(single_owner + exact expected F0.5, shifts 1.0/0.6). Fold-4 S1s never enter training.
"""
import glob, sys, time
import numpy as np, polars as pl, lightgbm as lgb
from v10env import WORK
from stack import FEATURES_D, PARAMS
from decide import dta_select, prior_shift, single_owner
from score import per_entity
import newfeats

t0 = time.time(); T = f"{WORK}/train"; n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
keep_neg = (pl.struct("s1", "t").hash(seed=5) % 100) < 10
df = pl.concat([pl.read_parquet(p) for p in sorted(glob.glob(f"{T}/feat_d/part_*.parquet"))[:n]])
df = newfeats.add(df)
tr = df.filter((pl.col("fold") != 4) & ((pl.col("label") == 1) | keep_neg))
ev = df.filter(pl.col("fold") == 4)
gt = pl.read_parquet(f"{T}/gt.parquet").join(ev.select("s1").unique(), on="s1", how="semi")
ids = ev.select("s1").unique()
print(f"train {tr.height:,} eval pairs {ev.height:,} S1 {ids.height:,} ({time.time()-t0:.0f}s)", flush=True)
for name, feats in (("base", list(FEATURES_D)), ("new", list(FEATURES_D) + newfeats.NEW)):
    w = np.where(tr["label"].to_numpy() == 1, 1.0, 10.0)
    va = tr.filter(pl.col("fold") == 3).sample(fraction=0.2, seed=2)
    b = lgb.train(PARAMS, lgb.Dataset(tr.filter(pl.col("fold") != 3).select(feats).to_numpy(), tr.filter(pl.col("fold") != 3)["label"].to_numpy(),
                  weight=w[(tr["fold"] != 3).to_numpy()], feature_name=feats), 1000,
                  valid_sets=[lgb.Dataset(va.select(feats).to_numpy(), va["label"].to_numpy(), weight=np.where(va["label"].to_numpy() == 1, 1.0, 10.0))],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    sc = ev.select("s1", "t").with_columns(p=pl.Series(b.predict(ev.select(feats).to_numpy())))
    res = []
    for s in (1.0, 0.6):
        pred = dta_select(single_owner(prior_shift(sc, s)).filter(pl.col("p") > 1e-4), 0.0).select("s1", "t")
        res.append(f"shift {s}: {per_entity(ids, pred, gt)['f'].mean():.5f}")
    gain = dict(zip(feats, b.feature_importance("gain")))
    top_new = {k: round(gain[k] / sum(gain.values()), 4) for k in newfeats.NEW} if name == "new" else {}
    print(f"{name}: iters {b.best_iteration} | " + " | ".join(res), top_new, f"({time.time()-t0:.0f}s)", flush=True)
