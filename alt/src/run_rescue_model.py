import json, time, polars as pl, lightgbm as lgb
from v10env import WORK
from matcher_b import apply_policy
from score import per_entity
import rescue_model as R
t0 = time.time(); T = f"{WORK}/train"
meta = json.load(open(f"{WORK}/models/matcher_c/meta.json"))
gt = pl.read_parquet(f"{T}/gt.parquet").with_columns(label=pl.lit(1, pl.UInt8))
folds = pl.read_parquet(f"{T}/folds.parquet", columns=["s1", "fold"])
s1 = pl.read_parquet(f"{T}/s1.parquet", columns=["s1", "country", "core", "core_key", "concat", "legal", "addr_blank"]).sort("s1")
tg = pl.read_parquet(f"{T}/tgt.parquet", columns=["t", "country", "src", "core", "core_key", "concat", "legal", "addr_blank", "is_domain"]).sort("t")
sc = pl.read_parquet(f"{T}/scores_c.parquet", columns=["s1", "t", "p"])
pred = apply_policy(sc.filter(pl.col("p") > 1e-4), meta["policy"])
best = sc.group_by("s1").agg(s_best_p=pl.col("p").max())
cand = R.candidates(s1, tg, pred, sc)
del sc
print(f"rescue candidates {cand.height:,} ({time.time()-t0:.0f}s)", flush=True)
x = R.features(cand, s1, tg, pred, best).join(folds, on="s1").join(gt, on=["s1", "t"], how="left").with_columns(pl.col("label").fill_null(0))
print(f"features ({time.time()-t0:.0f}s); positives {int(x['label'].sum()):,} of {x.height:,}", flush=True)
b = {"A": R.fit(x, [0, 1]), "B": R.fit(x, [2, 3])}
b["A"].save_model(f"{WORK}/rescue_A.txt"); b["B"].save_model(f"{WORK}/rescue_B.txt")
q = R.predict(b, x)
for part, ids in (("dev", folds.filter(pl.col("fold") != 4)), ("sealed", folds.filter(pl.col("fold") == 4))):
    base = per_entity(ids.select("s1"), pred, gt)["f"].mean()
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
        add = R.select(x, q, tau).join(ids.select("s1"), on="s1", how="semi")
        prec = add.join(gt, on=["s1", "t"], how="semi").height / max(add.height, 1)
        new = per_entity(ids.select("s1"), pl.concat([pred, add]), gt)["f"].mean()
        print(f"{part:6s} tau {tau}: add {add.height:,} precision {prec:.3f} macro {base:.5f} -> {new:.5f} ({new-base:+.5f})", flush=True)
