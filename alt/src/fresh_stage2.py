"""Fresh stage-2 model: one LightGBM on ALL dev folds (0-3) instead of v10's two half-data
cross-fit models. Early stopping on a held-out 20% of fold-3 S1s. Fold 4 never enters training.
Saves the booster; reports fold-4 macro F0.5 per country vs v10 stored scores (same S1s)."""
import glob, time
import numpy as np, polars as pl, lightgbm as lgb
from v10env import WORK
from stack import FEATURES_D, PARAMS
from decide import dta_select, prior_shift, single_owner
from score import per_entity

t0 = time.time(); T = f"{WORK}/train"; F = list(FEATURES_D)
parts = sorted(glob.glob(f"{T}/feat_d/part_*.parquet"))
keep_neg = (pl.struct("s1", "t").hash(seed=5) % 100) < 10
va_s1 = (pl.col("fold") == 3) & ((pl.col("s1").hash(seed=9) % 5) == 0)
tr_l, va_l, ev_l = [], [], []
for p in parts:
    x = pl.read_parquet(p)
    tr_l.append(x.filter((pl.col("fold") != 4) & ~va_s1 & ((pl.col("label") == 1) | keep_neg)))
    va_l.append(x.filter(va_s1 & ((pl.col("label") == 1) | keep_neg)))
    ev_l.append(x.filter(pl.col("fold") == 4))
tr, va, ev = pl.concat(tr_l), pl.concat(va_l), pl.concat(ev_l); del tr_l, va_l, ev_l
print(f"train {tr.height:,} val {va.height:,} eval pairs {ev.height:,} ({time.time()-t0:.0f}s)", flush=True)
w = lambda d: np.where(d["label"].to_numpy() == 1, 1.0, 10.0)
b = lgb.train(PARAMS, lgb.Dataset(tr.select(F).to_numpy(), tr["label"].to_numpy(), weight=w(tr), feature_name=F), 1500,
              valid_sets=[lgb.Dataset(va.select(F).to_numpy(), va["label"].to_numpy(), weight=w(va))], callbacks=[lgb.early_stopping(40, verbose=False)])
b.save_model(f"{WORK}/fresh_stage2.txt")
print(f"iters {b.best_iteration} ({time.time()-t0:.0f}s)", flush=True)
cty = pl.read_parquet(f"{T}/s1.parquet", columns=["s1", "country"])
ids = ev.select("s1").unique().join(cty, on="s1")
gt = pl.read_parquet(f"{T}/gt.parquet").join(ids, on="s1", how="semi")
new = ev.select("s1", "t").with_columns(p=pl.Series(b.predict(ev.select(F).to_numpy())))
old = pl.scan_parquet(f"{T}/scores_c.parquet").select("s1", "t", "p").collect().join(ids.select("s1"), on="s1", how="semi")
for s in (1.0, 0.8, 0.6, 0.45):
    row = []
    for name, sc in (("v10", old), ("fresh", new)):
        pred = dta_select(single_owner(prior_shift(sc, s)).filter(pl.col("p") > 1e-4), 0.0).select("s1", "t")
        e = per_entity(ids.select("s1"), pred, gt).join(ids, on="s1")
        row.append(f"{name} {e['f'].mean():.5f} " + str({k: round(v, 5) for k, v in e.group_by('country').agg(pl.col('f').mean()).iter_rows()}))
    print(f"shift {s}: " + " | ".join(row), flush=True)
