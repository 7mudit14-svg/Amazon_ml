"""LOCO grid for self-training: thresholds, weight, rounds (domain flag always on)."""
import runpy, sys, numpy as np, polars as pl
g = runpy.run_path("loco_selftrain.py", run_name="loco")  # reuses data + helpers (runs B0-B2 once)
fit, evaluate, F, us_tr, in_pool, in_ev, w, T = (g[k] for k in ("fit", "evaluate", "F", "us_tr", "in_pool", "in_ev", "w", "T"))
F2 = F + ["dom"]
g["us_va"] = g["us_va"].with_columns(dom=pl.lit(0.0, pl.Float32))
ev = in_ev.with_columns(dom=pl.lit(1.0, pl.Float32))
pool = in_pool.with_columns(dom=pl.lit(1.0, pl.Float32))
base = us_tr.with_columns(dom=pl.lit(0.0, pl.Float32))
b0 = g["b0"]
for hi, lo, wt, rounds in [(0.95, 0.05, 0.5, 1), (0.98, 0.02, 1.0, 1), (0.98, 0.02, 0.5, 2), (0.95, 0.05, 1.0, 2), (0.9, 0.05, 1.0, 3)]:
    model, feats = b0, F
    for r in range(rounds):
        q = model.predict(pool.select(feats).to_numpy())
        pp = pool.with_columns(q=pl.Series(q))
        rows = pl.concat([pp.filter((pl.col("q") >= hi) & (pl.col("num_match") == 1)).with_columns(label=pl.lit(1, pl.UInt8)),
                          pp.filter(pl.col("q") <= lo).with_columns(label=pl.lit(0, pl.UInt8))]).drop("q")
        tr = pl.concat([base, rows.select(base.columns)])
        model = fit(tr, F2, np.concatenate([w(base), wt * w(rows)])); feats = F2
    evaluate(f"hi={hi} lo={lo} w={wt} rounds={rounds}", model, F2, ev)
