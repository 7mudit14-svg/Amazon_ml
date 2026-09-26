"""Cluster/sibling rescue: an unclaimed target inherits the owner of its near-duplicate siblings.

A target t that the decision assigned to nobody (typically blank address or website-only name)
is given to S1 X when the confident claimed targets sharing t's normalized name key (same country)
all belong to exactly one S1 X (and at least `min_sib` of them). Chains (siblings owned by several
S1s) are skipped. Also checks the hard caps measured on train (<=5 S2 and <=6 S3 matches per S1).
"""
import polars as pl


def rescue(pred, tg, which="blank", min_sib=1, key="core_key"):
    """pred: s1, t (base decision). tg: t, country, core_key, concat, addr_blank, is_domain."""
    claimed = pred.join(tg.select("t", "country", k=key), on="t")
    own = claimed.filter(pl.col("k") != "").group_by("country", "k").agg(n_own=pl.col("s1").n_unique(), s1=pl.col("s1").first(), n_sib=pl.len())
    own = own.filter((pl.col("n_own") == 1) & (pl.col("n_sib") >= min_sib)).drop("n_own", "n_sib")
    cond = {"blank": pl.col("addr_blank"), "blank+domain": pl.col("addr_blank") | pl.col("is_domain"), "all": pl.lit(True)}[which]
    free = tg.filter(cond).join(pred.select("t").unique(), on="t", how="anti").select("t", "country", k=key)
    return free.join(own, on=["country", "k"]).select("s1", "t")


if __name__ == "__main__":
    import json
    from v10env import WORK
    from matcher_b import apply_policy
    from score import per_entity
    T = f"{WORK}/train"
    meta = json.load(open(f"{WORK}/models/matcher_c/meta.json"))
    gt = pl.read_parquet(f"{T}/gt.parquet"); folds = pl.read_parquet(f"{T}/folds.parquet", columns=["s1", "fold"])
    tg = pl.read_parquet(f"{T}/tgt.parquet", columns=["t", "country", "src", "core_key", "concat", "addr_blank", "is_domain"])
    sc = pl.read_parquet(f"{T}/scores_c.parquet", columns=["s1", "t", "p"])
    pred = apply_policy(sc.filter(pl.col("p") > 1e-4), meta["policy"]); del sc
    cnt = pred.join(tg.select("t", "src"), on="t").group_by("s1", "src").agg(n=pl.len())
    print("S1s over caps (S2>5 or S3>6):", cnt.filter(((pl.col("src") == 2) & (pl.col("n") > 5)) | ((pl.col("src") == 3) & (pl.col("n") > 6))).height)
    for part, ids in (("dev", folds.filter(pl.col("fold") != 4)), ("sealed", folds.filter(pl.col("fold") == 4))):
        base = per_entity(ids.select("s1"), pred, gt)["f"].mean()
        for which, ms, key in (("blank", 1, "core_key"), ("blank", 2, "core_key"), ("blank+domain", 2, "core_key"), ("blank", 2, "concat"), ("all", 2, "core_key")):
            add = rescue(pred, tg, which, ms, key).join(ids.select("s1"), on="s1", how="semi")
            prec = add.join(gt, on=["s1", "t"], how="semi").height / max(add.height, 1)
            new = per_entity(ids.select("s1"), pl.concat([pred, add]), gt)["f"].mean()
            print(f"{part:6s} {which:12s} min_sib={ms} key={key:8s}: add {add.height:,} prec {prec:.3f} macro {base:.5f} -> {new:.5f} ({new-base:+.5f})", flush=True)
