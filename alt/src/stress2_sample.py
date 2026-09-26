"""Stress-v2 sampler: a US train split at test density, written as a v10 'test' split.

Test US has 0.50x the S1s and 0.62x the targets of train US. Train is the same generator with
fewer S1 families, ~19% of whose S1s are hidden (their records stay as orphans). So: keep a
fraction f of US S1 families (S1 + its owned records) and the same fraction of unowned records,
then hide `hide` of the kept S1s. Rows keep v10's prepared (normalized) columns; indices are
re-densified so the unchanged v10 test path runs on it. Mapping and private ground truth are
written next to it; nothing from the labels enters the pipeline inputs.
"""
import argparse, os
import polars as pl
from v10env import WORK

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/home/user/work_s2")
ap.add_argument("--country", default="US")
ap.add_argument("--f", type=float, default=0.62)
ap.add_argument("--hide", type=float, default=0.19)
ap.add_argument("--seed", type=int, default=11)
a = ap.parse_args()
T = f"{WORK}/train"
out = f"{a.out}/test"; os.makedirs(out, exist_ok=True)

gt = pl.read_parquet(f"{T}/gt.parquet")
s1 = pl.scan_parquet(f"{T}/s1.parquet").filter(pl.col("country") == a.country).collect()
u = (pl.col("s1").hash(seed=a.seed) % 1_000_000).cast(pl.Float64) / 1e6
fam = s1.select("s1").with_columns(_u=u).filter(pl.col("_u") < a.f)
visible = fam.filter(pl.col("_u") >= a.f * a.hide).select("s1")        # hide the first `hide` share
owned_t = gt.join(fam.select("s1"), on="s1", how="semi").select("t")
all_owned = gt.select("t")
tg = pl.scan_parquet(f"{T}/tgt.parquet").filter(pl.col("country") == a.country).collect()
un = tg.select("t").join(all_owned, on="t", how="anti").with_columns(
    _u=(pl.col("t").hash(seed=a.seed + 1) % 1_000_000).cast(pl.Float64) / 1e6).filter(pl.col("_u") < a.f).select("t")
keep_t = pl.concat([owned_t, un]).unique().sort("t")

s1o = s1.join(visible, on="s1", how="semi").sort("s1").with_columns(orig_s1=pl.col("s1"))
s1o = s1o.with_columns(s1=pl.int_range(pl.len(), dtype=pl.UInt32))
tgo = tg.join(keep_t, on="t", how="semi").sort("t").with_columns(orig_t=pl.col("t"))
tgo = tgo.with_columns(t=pl.int_range(pl.len(), dtype=pl.UInt32))
s1o.drop("orig_s1").write_parquet(f"{out}/s1.parquet")
tgo.drop("orig_t").write_parquet(f"{out}/tgt.parquet")
for name, idc, m in (("s1_sk", "s1", s1o), ("tgt_sk", "t", tgo)):
    sk = pl.read_parquet(f"{T}/{name}.parquet").join(m.select(pl.col(f"orig_{idc}").alias(idc), new=idc), on=idc)
    sk.select(pl.col("new").alias(idc), "name_sk").sort(idc).write_parquet(f"{out}/{name}.parquet")
s1o.select("s1", "orig_s1").write_parquet(f"{a.out}/map_s1.parquet")
tgo.select("t", "orig_t").write_parquet(f"{a.out}/map_t.parquet")
print(f"{a.country} f={a.f} hide={a.hide}: families {fam.height:,}, visible S1 {s1o.height:,} "
      f"(train {s1.height:,}), targets {tgo.height:,} (train {tg.height:,}); per S1 {tgo.height/s1o.height:.2f}")
