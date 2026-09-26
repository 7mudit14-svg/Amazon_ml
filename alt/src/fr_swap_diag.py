"""Label-free check: share of confident pairs that swap a whole name word while keeping the house
number, by country (test), against the same statistic on train with labels."""
import polars as pl
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

def swap_stats(sc, s1, tg):
    x = sc.join(s1, on="s1").join(tg, on="t")
    a = pl.col("s_core").str.split(" ").list.set_difference(pl.lit([""]))
    b = pl.col("t_core").str.split(" ").list.set_difference(pl.lit([""]))
    x = x.with_columns(new=b.list.set_difference(a), gone=a.list.set_difference(b),
                       num_eq=(pl.col("s_nums").str.split(" ").list.first() == pl.col("t_nums").str.split(" ").list.first()).fill_null(False))
    x = x.with_columns(n_new=pl.col("new").list.len(), n_gone=pl.col("gone").list.len(),
                       w_new=pl.col("new").list.first().fill_null(""), w_gone=pl.col("gone").list.first().fill_null(""))
    d = process.cpdist(x["w_new"].to_list(), x["w_gone"].to_list(), scorer=Levenshtein.normalized_similarity, workers=-1)
    x = x.with_columns(sim=pl.Series(d))
    # whole-word substitution: one word replaced by a dissimilar word (not a typo)
    return x.with_columns(swap=(pl.col("n_new") >= 1) & (pl.col("n_gone") >= 1) & (pl.col("sim") < 0.5) & pl.col("num_eq"))

if __name__ == "__main__":
    W = "/home/user/work_test/test"
    sc = pl.read_parquet(f"{W}/scores_c.parquet", columns=["s1", "t", "p"]).filter(pl.col("p") > 0.5)
    s1 = pl.read_parquet(f"{W}/s1.parquet", columns=["s1", "country", "core", "addr_nums"]).rename({"core": "s_core", "addr_nums": "s_nums"})
    tg = pl.read_parquet(f"{W}/tgt.parquet", columns=["t", "core", "addr_nums"]).rename({"core": "t_core", "addr_nums": "t_nums"})
    x = swap_stats(sc, s1, tg)
    print("TEST p>0.5:", x.group_by("country").agg(n=pl.len(), swap_share=pl.col("swap").mean(), swap_hi=(pl.col("swap") & (pl.col("p") > 0.9)).mean()).sort("country"))
    T = "/home/user/work_v10/train"
    sc = pl.read_parquet(f"{T}/scores_c.parquet", columns=["s1", "t", "p"]).filter(pl.col("p") > 0.5)
    gt = pl.read_parquet(f"{T}/gt.parquet").with_columns(y=pl.lit(1))
    s1 = pl.read_parquet(f"{T}/s1.parquet", columns=["s1", "country", "core", "addr_nums"]).rename({"core": "s_core", "addr_nums": "s_nums"})
    tg = pl.read_parquet(f"{T}/tgt.parquet", columns=["t", "core", "addr_nums"]).rename({"core": "t_core", "addr_nums": "t_nums"})
    x = swap_stats(sc, s1, tg).join(gt, on=["s1", "t"], how="left").with_columns(pl.col("y").fill_null(0))
    print("TRAIN p>0.5:", x.group_by("country").agg(n=pl.len(), swap_share=pl.col("swap").mean(),
          true_rate_swap=pl.col("y").filter(pl.col("swap")).mean(), true_rate_all=pl.col("y").mean()).sort("country"))
