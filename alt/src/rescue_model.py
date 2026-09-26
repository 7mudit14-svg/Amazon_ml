"""Stage R: recover true matches that v10's blocking never retrieved (no retraining of v10).

Queries are targets that the base prediction assigned to nobody and whose name is the only usable
field (blank address) or a website/handle. For each, the top-K S1s of the same country by
character 3-gram TF-IDF cosine of the concatenated name (v10 `concat`) are retrieved; pairs already
in the base candidate list are dropped (the base model has judged them). A LightGBM model scores the
remaining pairs from name similarity, retrieval context and the S1's base prediction context.
It is cross-fitted like v10 (A: folds {0,1}, B: {2,3}; fold 4 and test get the mean). A pair is
added only above a threshold chosen on dev, with at most one S1 per target.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import lightgbm as lgb
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

K = 10
FEATS = ["cos", "rank", "cos_gap_next", "cos_best_other", "n_cos80", "n_cos90", "key_eq", "len_ratio",
         "core_ratio", "core_tset", "concat_jw", "legal_eq", "legal_conflict", "t_src3", "t_blank",
         "t_domain", "s_npred", "s_npred_src", "s_best_p", "s_name_rep", "s_blank", "t_n_ctok"]
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200,
          "lambda_l2": 10.0, "feature_fraction": 0.9, "verbosity": -1, "seed": 3, "deterministic": True,
          "force_row_wise": True}


def retrieve(s1: pl.DataFrame, q: pl.DataFrame) -> pl.DataFrame:
    """s1: s1, country, concat. q: t, country, concat. Top-K S1 per query by char-3gram cosine."""
    out = []
    for (c,), qc in q.group_by(["country"]):
        sc = s1.filter((pl.col("country") == c) & (pl.col("concat").str.len_chars() >= 3))
        if sc.height == 0 or qc.height == 0:
            continue
        vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=1, dtype=np.float32, sublinear_tf=True)
        A = vec.fit_transform(sc["concat"].to_list())
        B = vec.transform(qc["concat"].to_list())
        M = sp_matmul_topn(B, A.T.tocsr(), top_n=K, threshold=0.3, n_threads=4).tocoo()
        out.append(pl.DataFrame({"t": qc["t"].to_numpy()[M.row], "s1": sc["s1"].to_numpy()[M.col],
                                 "cos": M.data.astype(np.float32)}))
    return pl.concat(out) if out else pl.DataFrame(schema={"t": pl.UInt32, "s1": pl.UInt32, "cos": pl.Float32})


def features(pairs: pl.DataFrame, s1: pl.DataFrame, tg: pl.DataFrame, base_pred: pl.DataFrame,
             base_best: pl.DataFrame) -> pl.DataFrame:
    """pairs: t, s1, cos. s1/tg: prepared v10 frames (sorted by index). base_pred: s1, t predicted.
    base_best: s1, s_best_p (best base p per S1)."""
    x = pairs.with_columns(
        rank=pl.col("cos").rank("ordinal", descending=True).over("t").cast(pl.Float32),
        n_cos80=(pl.col("cos") > 0.8).sum().over("t").cast(pl.Float32),
        n_cos90=(pl.col("cos") > 0.9).sum().over("t").cast(pl.Float32),
    ).sort(["t", "cos"], descending=[False, True])
    x = x.with_columns(cos_gap_next=(pl.col("cos") - pl.col("cos").shift(-1).over("t")).fill_null(pl.col("cos")),
                       _mx=pl.col("cos").max().over("t"), _mx2=pl.col("cos").slice(1, 1).first().over("t"))
    x = x.with_columns(cos_best_other=pl.when(pl.col("rank") == 1).then(pl.col("_mx2")).otherwise(pl.col("_mx")).fill_null(0.0)).drop("_mx", "_mx2")
    S = s1[x["s1"].to_numpy()].select(s_core="core", s_key="core_key", s_concat="concat", s_legal="legal", s_blank="addr_blank")
    T = tg[x["t"].to_numpy()].select(t_core="core", t_key="core_key", t_concat="concat", t_legal="legal",
                                     t_blank="addr_blank", t_domain="is_domain", src="src")
    x = pl.concat([x, S, T], how="horizontal")
    def ratio(a, b, scorer, scale):
        r = process.cpdist(x[a].to_list(), x[b].to_list(), scorer=scorer, workers=-1, dtype=np.float32) / scale
        return pl.Series(r)
    x = x.with_columns(
        key_eq=(pl.col("s_key") == pl.col("t_key")) & (pl.col("s_key") != ""),
        len_ratio=pl.min_horizontal(pl.col("s_concat").str.len_chars(), pl.col("t_concat").str.len_chars())
        / pl.max_horizontal(pl.col("s_concat").str.len_chars(), pl.col("t_concat").str.len_chars()).clip(1, None),
        legal_eq=(pl.col("s_legal") == pl.col("t_legal")) & (pl.col("s_legal") != ""),
        legal_conflict=(pl.col("s_legal") != "") & (pl.col("t_legal") != "") & (pl.col("s_legal") != pl.col("t_legal")),
        t_src3=pl.col("src") == 3,
        t_n_ctok=pl.col("t_core").str.count_matches(" ") + 1,
    ).with_columns(core_ratio=ratio("s_core", "t_core", fuzz.ratio, 100.0),
                   core_tset=ratio("s_core", "t_core", fuzz.token_set_ratio, 100.0),
                   concat_jw=ratio("s_concat", "t_concat", JaroWinkler.normalized_similarity, 1.0))
    npred = base_pred.group_by("s1").agg(s_npred=pl.len())
    tsrc = tg.select("t", "src")
    npsrc = base_pred.join(tsrc, on="t").group_by("s1", "src").agg(s_npred_src=pl.len())
    rep = s1.filter(pl.col("core_key") != "").group_by("country", "core_key").agg(s_name_rep=pl.len())
    x = (x.join(npred, on="s1", how="left").join(npsrc, on=["s1", "src"], how="left").join(base_best, on="s1", how="left")
         .join(s1.select("s1", "country"), on="s1").join(rep, left_on=["country", "s_key"], right_on=["country", "core_key"], how="left")
         .with_columns(pl.col("s_npred", "s_npred_src", "s_name_rep").fill_null(0), pl.col("s_best_p").fill_null(0.0)))
    return x.select(["t", "s1"] + [pl.col(f).cast(pl.Float32) for f in FEATS])


def candidates(s1: pl.DataFrame, tg: pl.DataFrame, base_pred: pl.DataFrame, base_cand: pl.DataFrame) -> pl.DataFrame:
    """Unclaimed blank-address / domain targets -> top-K S1 pairs not in the base candidate set."""
    claimed = base_pred.select("t").unique()
    q = (tg.filter((pl.col("addr_blank") | pl.col("is_domain")) & (pl.col("concat").str.len_chars() >= 3))
         .join(claimed, on="t", how="anti").select("t", "country", "concat"))
    pairs = retrieve(s1.select("s1", "country", "concat"), q)
    return pairs.join(base_cand.select("s1", "t"), on=["s1", "t"], how="anti")


def fit(x: pl.DataFrame, folds: list[int]) -> lgb.Booster:
    tr = x.filter(pl.col("fold").is_in(folds))
    return lgb.train(PARAMS, lgb.Dataset(tr.select(FEATS).to_numpy(), tr["label"].to_numpy(), feature_name=FEATS),
                     num_boost_round=400)


def predict(b: dict, x: pl.DataFrame) -> np.ndarray:
    m = x.select(FEATS).to_numpy()
    pa, pb = b["A"].predict(m), b["B"].predict(m)
    if "fold" not in x.columns:
        return 0.5 * (pa + pb)
    f = x["fold"].to_numpy()
    return np.where(np.isin(f, [2, 3]), pa, np.where(np.isin(f, [0, 1]), pb, 0.5 * (pa + pb)))


def select(x: pl.DataFrame, q: np.ndarray, tau: float) -> pl.DataFrame:
    """At most one S1 per target: the best-scoring pair, if above tau."""
    return (x.select("s1", "t").with_columns(q=pl.Series(q)).filter(pl.col("q") > tau)
            .sort("q", descending=True).unique("t", keep="first").select("s1", "t"))
