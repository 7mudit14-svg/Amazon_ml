"""Post-decision rescue of unclaimed targets (no retraining).

Blank-address targets are ~98% owned in train, and 17% of US blank-address true pairs are never
retrieved by v10's blocking. A target that no S1 was assigned is given to an S1 when its
normalized name key (v10 core_key, order-insensitive) matches exactly one S1 of the same country.
Variants: which targets qualify (blank address only / also website-like names).
"""
import polars as pl


def rescue(pred: pl.DataFrame, s1: pl.DataFrame, tg: pl.DataFrame, which: str = "blank",
           max_s1_len: int | None = None) -> pl.DataFrame:
    """pred: s1, t. s1: s1, country, core_key. tg: t, country, core_key, addr_blank, is_domain.
    Returns the added (s1, t) pairs."""
    claimed = pred.select("t").unique()
    cond = pl.col("addr_blank") if which == "blank" else (pl.col("addr_blank") | pl.col("is_domain"))
    free = tg.filter(cond & (pl.col("core_key") != "")).join(claimed, on="t", how="anti")
    uniq = s1.filter(pl.col("core_key") != "").group_by("country", "core_key").agg(
        n=pl.len(), s1=pl.col("s1").first()).filter(pl.col("n") == 1).drop("n")
    add = free.join(uniq, on=["country", "core_key"]).select("s1", "t")
    if max_s1_len is not None:     # only S1s whose current list is short
        cnt = pred.group_by("s1").agg(k=pl.len())
        add = add.join(cnt, on="s1", how="left").filter(pl.col("k").fill_null(0) <= max_s1_len).select("s1", "t")
    return add
