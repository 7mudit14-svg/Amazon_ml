"""Evaluate ownership layer vs v10 policy on dev (OOF) and sealed fold 4."""
import json, time
import numpy as np, polars as pl
from v10env import WORK, s1_train, tgt_train
from decide import dta_select, prior_shift, single_owner
from score import per_entity
import ownership as own

t0 = time.time()
gt = pl.read_parquet(f"{WORK}/train/gt.parquet").with_columns(label=pl.lit(1, pl.UInt8))
folds = pl.read_parquet(f"{WORK}/train/folds.parquet", columns=["s1", "fold"])
s1 = s1_train().join(folds, on="s1")
ta = tgt_train().select("t", a="business_address")
tb = ta.select("t", t_blank=(pl.col("a").is_null() | pl.col("a").str.to_lowercase().is_in(["none", "null", "n/a", ""])).cast(pl.Float32))
sc = pl.read_parquet(f"{WORK}/train/scores_c.parquet", columns=["s1", "t", "p"]).join(tb, on="t", how="left")
x = own.features(sc).join(folds, on="s1").join(gt, on=["s1", "t"], how="left").with_columns(pl.col("label").fill_null(0))
print(f"rows {x.height:,} ({time.time()-t0:.0f}s)")
b = {"A": own.fit(x, [0, 1]), "B": own.fit(x, [2, 3])}
x = x.with_columns(q=pl.Series(own.predict(b, x)))
print(f"fitted ({time.time()-t0:.0f}s)")
b["A"].save_model(f"{WORK}/owner_A.txt"); b["B"].save_model(f"{WORK}/owner_B.txt")

def macro(ids, pred): return float(per_entity(ids.select("s1"), pred, gt)["f"].mean())
dev = s1.filter(pl.col("fold") != 4); sealed = s1.filter(pl.col("fold") == 4)
def run(pairs, ids, name, shift):
    s = pairs.join(ids.select("s1"), on="s1", how="semi")
    s = prior_shift(s, shift)
    if name == "own": s = single_owner(s)
    return dta_select(s.filter(pl.col("p") > 1e-4), 0.0).select("s1", "t")

base = sc.select("s1", "t", "p")
new = x.select("s1", "t", p="q")
res = []
for shift in [1.0, 0.8, 0.6, 0.45]:
    for label, pairs, pol in [("v10 dta_own", base, "own"), ("owner-model dta", new, "dta"), ("owner-model +own", new, "own")]:
        # dev entities only need their own + competitors' claims; single_owner needs all claims -> compute on full then filter
        s = prior_shift(pairs, shift)
        if pol == "own": s = single_owner(s)
        pd_ = dta_select(s.filter(pl.col("p") > 1e-4).join(dev.select("s1"), on="s1", how="semi"), 0.0).select("s1", "t")
        f = macro(dev, pd_)
        res.append((label, shift, f)); print(f"dev {label:18s} shift {shift}: {f:.5f} ({time.time()-t0:.0f}s)", flush=True)
best = {}
for label in {r[0] for r in res}:
    best[label] = max((r for r in res if r[0] == label), key=lambda r: r[2])
for label, (_, shift, f) in best.items():
    pairs = base if label.startswith("v10") else new
    s = prior_shift(pairs, shift)
    if label.endswith("own"): s = single_owner(s)
    pd_ = dta_select(s.filter(pl.col("p") > 1e-4).join(sealed.select("s1"), on="s1", how="semi"), 0.0).select("s1", "t")
    ent = per_entity(sealed.select("s1"), pd_, gt).join(sealed, on="s1")
    print(f"SEALED {label:18s} (dev-best shift {shift}): {ent['f'].mean():.5f}  ",
          dict(ent.group_by("country").agg(pl.col("f").mean()).iter_rows()))
