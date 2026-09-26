"""W2 gate: does stage-2 self-training on an unlabelled country help? (LOCO proxy for France)

Stage 2 is refit on US labelled dev rows only. India plays the unseen country: its dev-fold rows
are used without labels (pseudo-labels from the US-only model), and India fold 4 is scored with
its real labels. Variants: B0 US only; B1 + India pseudo-labels; B2 = B1 + a domain flag feature.
Caveat: the stage-1 probability inside feat_d was fitted on US+India, so this proxy is optimistic
about the base model; it still isolates what self-training adds on top of it.
"""
import glob, sys, time
import numpy as np, polars as pl, lightgbm as lgb
from v10env import WORK
from stack import FEATURES_D, PARAMS
from decide import dta_select, prior_shift, single_owner
from score import per_entity

t0 = time.time(); T = f"{WORK}/train"
parts = sorted(glob.glob(f"{T}/feat_d/part_*.parquet"))
cty = pl.read_parquet(f"{T}/s1.parquet", columns=["s1", "country"])
keep_neg = (pl.struct("s1", "t").hash(seed=5) % 100) < 20
df = pl.concat([pl.read_parquet(p).filter((pl.col("label") == 1) | keep_neg | (pl.col("fold") == 4)) for p in parts]).join(cty, on="s1")
print(f"rows {df.height:,} ({time.time()-t0:.0f}s)", flush=True)
us_tr = df.filter((pl.col("country") == "US") & (pl.col("fold") != 4))
us_va = df.filter((pl.col("country") == "US") & (pl.col("fold") == 4)).sample(fraction=0.3, seed=1)
in_pool = df.filter((pl.col("country") == "India") & (pl.col("fold") != 4))
in_ev = df.filter((pl.col("country") == "India") & (pl.col("fold") == 4))
gt = pl.read_parquet(f"{T}/gt.parquet").join(in_ev.select("s1").unique(), on="s1", how="semi")

def w(d): return np.where(d["label"].to_numpy() == 1, 1.0, 5.0)

def fit(tr, feats, weights):
    dtr = lgb.Dataset(tr.select(feats).to_numpy(), tr["label"].to_numpy(), weight=weights, feature_name=feats)
    dva = lgb.Dataset(us_va.select(feats).to_numpy(), us_va["label"].to_numpy(), weight=w(us_va), reference=dtr)
    return lgb.train(PARAMS, dtr, num_boost_round=600, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])

def evaluate(name, b, feats, ev):
    p = b.predict(ev.select(feats).to_numpy())
    sc = ev.select("s1", "t").with_columns(p=pl.Series(p))
    out = []
    for s in (1.0, 0.6, 0.45, 0.3):
        s_ = single_owner(prior_shift(sc, s))
        pred = dta_select(s_.filter(pl.col("p") > 1e-4), 0.0).select("s1", "t")
        out.append(f"shift {s}: {per_entity(in_ev.select('s1').unique(), pred, gt)['f'].mean():.5f}")
    print(f"{name} (iters {b.best_iteration}): " + " | ".join(out), f"({time.time()-t0:.0f}s)", flush=True)
    return p

F = list(FEATURES_D)
b0 = fit(us_tr, F, w(us_tr)); evaluate("B0 US-only", b0, F, in_ev)
q = b0.predict(in_pool.select(F).to_numpy())
pool = in_pool.with_columns(q=pl.Series(q))
pos = pool.filter((pl.col("q") >= 0.98) & (pl.col("num_match") == 1)).with_columns(label=pl.lit(1, pl.UInt8))
neg = pool.filter(pl.col("q") <= 0.02).with_columns(label=pl.lit(0, pl.UInt8))
pl_rows = pl.concat([pos, neg]).drop("q")
true_prec = pos.join(pl.read_parquet(f"{T}/gt.parquet"), on=["s1", "t"], how="semi").height / max(pos.height, 1)
print(f"pseudo-labels: pos {pos.height:,} (true precision {true_prec:.4f}), neg {neg.height:,}", flush=True)
tr1 = pl.concat([us_tr, pl_rows.select(us_tr.columns)])
wt1 = np.concatenate([w(us_tr), 0.5 * w(pl_rows)])
b1 = fit(tr1, F, wt1); evaluate("B1 +India pseudo", b1, F, in_ev)
F2 = F + ["dom"]
tr2 = tr1.with_columns(dom=pl.Series(np.r_[np.zeros(us_tr.height), np.ones(pl_rows.height)]).cast(pl.Float32))
us_va = us_va.with_columns(dom=pl.lit(0.0, pl.Float32))
b2 = fit(tr2, F2, wt1); evaluate("B2 +pseudo +domain flag", b2, F2, in_ev.with_columns(dom=pl.lit(1.0, pl.Float32)))
