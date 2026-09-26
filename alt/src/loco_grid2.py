"""LOCO variants of the self-training recipe (domain flag on, 1 round); baseline best = 0.9447."""
import runpy, numpy as np, polars as pl
g = runpy.run_path("loco_selftrain.py", run_name="loco")
fit, evaluate, F, us_tr, in_pool, in_ev, w = (g[k] for k in ("fit", "evaluate", "F", "us_tr", "in_pool", "in_ev", "w"))
F2 = F + ["dom"]
g["us_va"] = g["us_va"].with_columns(dom=pl.lit(0.0, pl.Float32))
ev = in_ev.with_columns(dom=pl.lit(1.0, pl.Float32))
base = us_tr.with_columns(dom=pl.lit(0.0, pl.Float32))
q = g["b0"].predict(in_pool.select(F).to_numpy())
pp = in_pool.with_columns(q=pl.Series(q), dom=pl.lit(1.0, pl.Float32),
                          r=pl.Series(q).rank("ordinal", descending=True).over(in_pool["s1"]) if False else pl.lit(0))
pp = pp.with_columns(r=pl.col("q").rank("ordinal", descending=True).over("s1"))
cons = (((pl.col("q") >= 0.98) | ((pl.col("q") >= 0.9) & (pl.col("sup_num_p") > 0.5))) & (pl.col("num_match") == 1))
variants = {
    "g) consistent + w=2": (cons, pl.col("q") <= 0.02, 2.0),
    "h) consistent + w=3": (cons, pl.col("q") <= 0.02, 3.0),
    "i) ref w=3": ((pl.col("q") >= 0.98) & (pl.col("num_match") == 1), pl.col("q") <= 0.02, 3.0),
}
for name, (pos_c, neg_c, wt) in variants.items():
    rows = pl.concat([pp.filter(pos_c).with_columns(label=pl.lit(1, pl.UInt8)), pp.filter(neg_c).with_columns(label=pl.lit(0, pl.UInt8))])
    tr = pl.concat([base, rows.select(base.columns)])
    m = fit(tr, F2, np.concatenate([w(base), wt * w(rows)]))
    evaluate(f"V {name}", m, F2, ev)
