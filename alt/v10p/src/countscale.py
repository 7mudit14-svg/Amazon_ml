"""Pool-size normalisation of split-dependent counts (off unless BER_COUNT_SCALE=1).

Counts such as the ranker's union size, block sizes, same-name counts, claimants per target and
name-token document frequencies grow with the number of records in the split. The models were
fitted on train pools (US 1.32M S1 / 6.19M targets, India 0.88M / 4.13M); test US has half the
S1s and 62% of the targets. With scaling on, each count is multiplied by
train_pool(country) / split_pool(country) (S1 pool for S1 counts, target pool for target counts),
clipped to [0.5, 3], so the models see train-scale values. Unseen countries use the US train
pool. Train pool sizes are counted from the train files; nothing label-dependent is used.
"""
from __future__ import annotations

import os

import polars as pl

TRAIN_POOL = {"US": (1_323_633, 6_186_873), "India": (883_188, 4_133_346)}
REF = "US"
CLIP = (0.5, 3.0)


def enabled() -> bool:
    return os.environ.get("BER_COUNT_SCALE") == "1"


def factors(s1: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """country, s_scale, t_scale from the split's own pool sizes (frames need a country column)."""
    ns = s1.group_by("country").agg(n_s=pl.len())
    nt = tg.group_by("country").agg(n_t=pl.len())
    rows = []
    for c, n_s, n_t in ns.join(nt, on="country", how="full", coalesce=True).fill_null(1).iter_rows():
        base_s, base_t = TRAIN_POOL.get(c, TRAIN_POOL[REF])
        rows.append((c, min(max(base_s / n_s, CLIP[0]), CLIP[1]), min(max(base_t / n_t, CLIP[0]), CLIP[1])))
    return pl.DataFrame(rows, schema={"country": pl.String, "s_scale": pl.Float64, "t_scale": pl.Float64}, orient="row")
