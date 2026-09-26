"""Step 2: target-level ownership model replacing decide.single_owner.

v10's stage 2 already sees target competition (c_gap, c_other_max), then single_owner divides
each claim again by 1 + sum of competitors' odds, assuming independent claims. Here a small
cross-fitted LightGBM re-scores every claim from its competition context (all S1s claiming the
same target) and its own list context, trained on dev-fold out-of-fold stage-2 scores; the
exact expected-F0.5 set selection (decide.dta_select) is unchanged.
Groups follow v10: model A fits folds {0,1} and scores {2,3}; B the reverse; fold 4 and test
get the mean.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import lightgbm as lgb

FEATS = ["lp", "rank_t", "n_claim", "n_claim10", "lp_other_max", "gap_other", "own_p", "sum_odds_other",
         "rank_s", "s_n10", "s_sum", "s_gap_best", "t_blank"]
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 2000,
          "lambda_l2": 10.0, "feature_fraction": 0.9, "verbosity": -1, "seed": 7, "deterministic": True,
          "force_row_wise": True}
P_MIN = 1e-3


def features(sc: pl.DataFrame) -> pl.DataFrame:
    """sc: s1, t, p (stage-2 calibrated), t_blank. Rows with p <= P_MIN are dropped."""
    x = sc.filter(pl.col("p") > P_MIN)
    pc = pl.col("p").clip(1e-6, 1 - 1e-6)
    odds = pc / (1 - pc)
    x = x.with_columns(lp=(odds).log(), _o=odds)
    x = x.with_columns(
        rank_t=pl.col("p").rank("ordinal", descending=True).over("t").cast(pl.Float32),
        n_claim=pl.len().over("t").cast(pl.Float32),
        n_claim10=(pl.col("p") > 0.1).sum().over("t").cast(pl.Float32),
        sum_odds_other=pl.col("_o").sum().over("t") - pl.col("_o"),
        rank_s=pl.col("p").rank("ordinal", descending=True).over("s1").cast(pl.Float32),
        s_n10=(pl.col("p") > 0.1).sum().over("s1").cast(pl.Float32),
        s_sum=pl.col("p").sum().over("s1"),
        s_gap_best=pl.col("p") - pl.col("p").max().over("s1"),
    )
    # best competing claim on the same target
    top2 = x.sort("p", descending=True).group_by("t", maintain_order=True).agg(
        p1=pl.col("p").first(), p2=pl.col("p").slice(1, 1).first())
    x = x.join(top2, on="t", how="left").with_columns(pl.col("p2").fill_null(0.0))
    other = pl.when(pl.col("rank_t") == 1).then(pl.col("p2")).otherwise(pl.col("p1"))
    x = x.with_columns(
        lp_other_max=(other.clip(1e-6, 1 - 1e-6) / (1 - other.clip(1e-6, 1 - 1e-6))).log(),
        gap_other=pl.col("p") - other,
        own_p=pl.col("_o") / (1 + pl.col("_o") + pl.col("sum_odds_other")),
    )
    return x.drop("_o", "p1", "p2")


def fit(x: pl.DataFrame, folds: list[int]) -> lgb.Booster:
    tr = x.filter(pl.col("fold").is_in(folds))
    d = lgb.Dataset(tr.select(FEATS).to_numpy(), tr["label"].to_numpy(), feature_name=FEATS)
    return lgb.train(PARAMS, d, num_boost_round=300)


def predict(boosters: dict, x: pl.DataFrame) -> np.ndarray:
    m = x.select(FEATS).to_numpy()
    pa, pb = boosters["A"].predict(m), boosters["B"].predict(m)
    if "fold" not in x.columns:
        return 0.5 * (pa + pb)
    f = x["fold"].to_numpy()
    return np.where(np.isin(f, [2, 3]), pa, np.where(np.isin(f, [0, 1]), pb, 0.5 * (pa + pb)))
