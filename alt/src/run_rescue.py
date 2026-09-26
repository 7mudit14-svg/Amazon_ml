import json, polars as pl
from v10env import WORK
from matcher_b import apply_policy
from score import per_entity
from rescue import rescue
T = f"{WORK}/train"
meta = json.load(open(f"{WORK}/models/matcher_c/meta.json"))
gt = pl.read_parquet(f"{T}/gt.parquet")
folds = pl.read_parquet(f"{T}/folds.parquet", columns=["s1", "fold"])
s1 = pl.read_parquet(f"{T}/s1.parquet", columns=["s1", "country", "core_key"])
tg = pl.read_parquet(f"{T}/tgt.parquet", columns=["t", "country", "core_key", "addr_blank", "is_domain"])
sc = pl.read_parquet(f"{T}/scores_c.parquet", columns=["s1", "t", "p"])
pred = apply_policy(sc.filter(pl.col("p") > 1e-4), meta["policy"])
del sc
for part, ids in (("dev", folds.filter(pl.col("fold") != 4)), ("sealed", folds.filter(pl.col("fold") == 4))):
    base = per_entity(ids.select("s1"), pred, gt).join(s1.select("s1", "country"), on="s1")
    for which in ("blank", "blank+domain"):
        for mx in (None, 0):
            add = rescue(pred, s1, tg, which, mx)
            a_e = add.join(ids.select("s1"), on="s1", how="semi")
            prec = a_e.join(gt, on=["s1", "t"], how="semi").height / max(a_e.height, 1)
            new = per_entity(ids.select("s1"), pl.concat([pred, add]), gt).join(s1.select("s1", "country"), on="s1")
            d = new["f"].mean() - base["f"].mean()
            byc = new.group_by("country").agg(pl.col("f").mean()).join(base.group_by("country").agg(b=pl.col("f").mean()), on="country")
            print(f"{part:6s} {which:13s} maxlen={mx}: added {a_e.height:,} pairs, precision {prec:.3f}, macro {base['f'].mean():.5f} -> {new['f'].mean():.5f} ({d:+.5f}) ",
                  {r[0]: round(r[1]-r[2], 5) for r in byc.iter_rows()}, flush=True)
