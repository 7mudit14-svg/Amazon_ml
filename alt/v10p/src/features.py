"""Plan A pair features for every candidate pair.

Set overlaps and flags are vectorized in polars; string ratios use rapidfuzz's
multi-threaded element-wise cpdist. Rank features are computed per S1 over the full
candidate list (each candidate part holds complete S1 lists).
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from blocking import PASS_COLS
from config import Config
from prepare import load

S1_COLS = ["s1", "country", "core", "core_key", "legal", "concat", "initials",
           "addr_nums", "addr_toks", "addr_blank"]
T_COLS = ["t", "src", "core", "core_key", "legal", "concat", "acro_self", "domain",
          "is_domain", "script", "addr_nums", "addr_toks", "addr_blank"]
FEATURES = [
    "name_jac", "name_exact", "name_ratio", "name_tset", "concat_lvl", "concat_jw", "acro",
    "legal_match", "legal_conflict",
    "addr_jac", "addr_tset", "num_match", "num_conflict", "t_blank", "s_blank",
    "log_name_tdf", "log_name_rep",
    "rare_idf", "rare_n", "addr_n", "f_exact", "f_concat",
    "src3", "t_indic", "t_domain",
    "cheap_rank", "cheap_rel", "log_n_cand",
]
# Stage 3 (variant "c", ranked candidates): ranker score, rare-address-word overlap,
# extra string views, near-miss house numbers and competition for the same target.
FEATURES_C = FEATURES + [
    "rk_p", "a2_n", "a2_idf", "name_pratio", "name_jw", "addr_ratio",
    "num_first_eq", "num_near", "num_suffix", "t_n_s1", "t_best_gap", "t_is_best",
    # Stage 5: cross-script name skeletons (normalize.name_skeleton) and the ranker's skeleton pass
    "sk_n", "sk_idf", "sk_tset", "sk_jw",
]
CAND_DIR = {"a": "cand_a", "b": "cand_b", "c": "cand_b"}


def _toks(col: str) -> pl.Expr:
    return pl.col(col).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def _inter(a: str, b: str) -> pl.Expr:
    return pl.col(a).list.set_intersection(pl.col(b)).list.len()


def _jac(a: str, b: str) -> pl.Expr:
    union = pl.col(a).list.set_union(pl.col(b)).list.len()
    return pl.when(union > 0).then(_inter(a, b) / union).otherwise(0.0)


def _prefix_levels(a: str, b: str) -> pl.Expr:
    total = pl.lit(0, pl.Int8)
    for length in (4, 6, 8, 12):
        ok = (
            (pl.col(a).str.len_chars() >= length)
            & (pl.col(b).str.len_chars() >= length)
            & (pl.col(a).str.slice(0, length) == pl.col(b).str.slice(0, length))
        )
        total = total + ok.cast(pl.Int8)
    return total


def _ratio(df: pl.DataFrame, a: str, b: str, scorer, scale: float) -> np.ndarray:
    left, right = df[a].to_list(), df[b].to_list()
    out = process.cpdist(left, right, scorer=scorer, workers=-1, dtype=np.float32) / scale
    empty = ((df[a] == "") | (df[b] == "")).to_numpy()
    out[empty] = 0.0
    return out


def name_frequencies(s1: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """How many same-country targets / S1 records carry exactly this core name."""
    tdf = tg.filter(pl.col("core_key") != "").group_by("country", "core_key").agg(name_tdf=pl.len())
    rep = s1.filter(pl.col("core_key") != "").group_by("country", "core_key").agg(name_rep=pl.len())
    return (
        s1.join(tdf, on=["country", "core_key"], how="left")
        .join(rep, on=["country", "core_key"], how="left")
        .with_columns(pl.col("name_tdf", "name_rep").fill_null(0))
        .pipe(_scale_names, s1, tg)
        .sort("s1")
    )


def _scale_names(df: pl.DataFrame, s1: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    import countscale
    if not countscale.enabled():
        return df
    f = countscale.factors(s1, tg)
    return (df.join(f, on="country", how="left")
            .with_columns(name_tdf=pl.col("name_tdf") * pl.col("t_scale"), name_rep=pl.col("name_rep") * pl.col("s_scale"))
            .drop("s_scale", "t_scale"))


def _near_numbers(df: pl.DataFrame) -> np.ndarray:
    """First house numbers that differ by exactly one edit (9914 vs 914, 103 vs 303)."""
    left, right = df["_s0"].to_list(), df["_t0"].to_list()
    dist = process.cpdist(left, right, scorer=Levenshtein.distance, workers=-1, dtype=np.int32)
    empty = ((df["_s0"] == "") | (df["_t0"] == "")).to_numpy()
    return ((dist == 1) & ~empty).astype(np.float32)


def _extended(df: pl.DataFrame) -> pl.DataFrame:
    s0 = pl.col("s_num").list.first()
    t0 = pl.col("t_num").list.first()
    df = df.with_columns(
        rk_p=pl.col("cheap"),
        num_first_eq=(s0 == t0).fill_null(False),
        num_suffix=((s0 != t0) & (s0.str.ends_with(t0) | t0.str.ends_with(s0))).fill_null(False),
        t_n_s1=pl.col("t_n_s1").cast(pl.Float32).log1p(),
        t_best_gap=pl.col("cheap") - pl.col("t_max_rk"),
        t_is_best=pl.col("cheap") >= pl.col("t_max_rk"),
        _s0=s0.fill_null(""),
        _t0=t0.fill_null(""),
    )
    return df.with_columns(
        name_pratio=pl.Series(_ratio(df, "s_core", "t_core", fuzz.partial_ratio, 100.0)),
        name_jw=pl.Series(_ratio(df, "s_core", "t_core", JaroWinkler.normalized_similarity, 1.0)),
        addr_ratio=pl.Series(_ratio(df, "s_addr_toks", "t_addr_toks", fuzz.ratio, 100.0)),
        num_near=pl.Series(_near_numbers(df)),
        sk_tset=pl.Series(_ratio(df, "s_name_sk", "t_name_sk", fuzz.token_set_ratio, 100.0)),
        sk_jw=pl.Series(_ratio(df, "s_name_sk", "t_name_sk", JaroWinkler.normalized_similarity, 1.0)),
    )


def competition(cfg: Config, split: str) -> pl.DataFrame:
    """Per target: how many S1s list it among their ranked candidates, and the best ranker score."""
    parts = sorted((cfg.split_dir(split) / "cand_b").glob("part_*.parquet"))
    c = pl.concat([
        pl.read_parquet(p, columns=["t", "cheap", "rrank"]).filter(pl.col("rrank") <= cfg.rk_cand_k).select("t", "cheap")
        for p in parts
    ])
    return c.group_by("t").agg(t_n_s1=pl.len(), t_max_rk=pl.col("cheap").max())


def pair_features(cand: pl.DataFrame, s1: pl.DataFrame, tg: pl.DataFrame, extended: bool = False) -> pl.DataFrame:
    """cand: s1, t, pass columns, cheap, rank features. s1/tg must be sorted by index."""
    left = s1[cand["s1"].to_numpy()].select(pl.all().name.prefix("s_")).drop("s_s1")
    right = tg[cand["t"].to_numpy()].select(pl.all().name.prefix("t_")).drop("t_t")
    df = pl.concat([cand, left, right], how="horizontal")
    df = df.with_columns(
        s_ctok=_toks("s_core"), t_ctok=_toks("t_core"),
        s_atok=_toks("s_addr_toks"), t_atok=_toks("t_addr_toks"),
        s_num=_toks("s_addr_nums"), t_num=_toks("t_addr_nums"),
        s_leg=_toks("s_legal"), t_leg=_toks("t_legal"),
    )
    num_inter = _inter("s_num", "t_num")
    leg_inter = _inter("s_leg", "t_leg")
    df = df.with_columns(
        name_jac=_jac("s_ctok", "t_ctok"),
        name_exact=(pl.col("s_core_key") == pl.col("t_core_key")) & (pl.col("s_core_key") != ""),
        concat_lvl=pl.max_horizontal(_prefix_levels("s_concat", "t_concat"), _prefix_levels("s_concat", "t_domain")),
        acro=(pl.col("t_acro_self") != "") & (pl.col("t_acro_self") == pl.col("s_initials")),
        legal_match=leg_inter > 0,
        legal_conflict=(pl.col("s_leg").list.len() > 0) & (pl.col("t_leg").list.len() > 0) & (leg_inter == 0),
        addr_jac=_jac("s_atok", "t_atok"),
        num_match=num_inter > 0,
        num_conflict=(pl.col("s_num").list.len() > 0) & (pl.col("t_num").list.len() > 0) & (num_inter == 0),
        t_blank=pl.col("t_addr_blank"),
        s_blank=pl.col("s_addr_blank"),
        log_name_tdf=pl.col("s_name_tdf").log1p(),
        log_name_rep=pl.col("s_name_rep").log1p(),
        src3=pl.col("t_src") == 3,
        t_indic=pl.col("t_script").is_in(["indic", "mixed"]),
        t_domain=pl.col("t_is_domain"),
    )
    df = df.with_columns(
        name_ratio=pl.Series(_ratio(df, "s_core", "t_core", fuzz.ratio, 100.0)),
        name_tset=pl.Series(_ratio(df, "s_core", "t_core", fuzz.token_set_ratio, 100.0)),
        concat_jw=pl.Series(_ratio(df, "s_concat", "t_concat", JaroWinkler.normalized_similarity, 1.0)),
        addr_tset=pl.Series(_ratio(df, "s_addr_toks", "t_addr_toks", fuzz.token_set_ratio, 100.0)),
    )
    feats = FEATURES
    if extended:
        df = _extended(df)
        feats = FEATURES_C
    keep = ["s1", "t"] + [c for c in ("label", "fold") if c in df.columns]
    return df.select(keep + [pl.col(f).cast(pl.Float32) for f in feats])


def build_features(cfg: Config, split: str, log=print, variant: str = "a") -> None:
    """variant "a": Plan A candidates; "b": ranked candidates cut to the top rk_cand_k per S1."""
    t0 = time.time()
    s1 = name_frequencies(load(cfg, split, "s1", S1_COLS), load(cfg, split, "tgt", ["t", "country", "core_key"]))
    tg = load(cfg, split, "tgt", T_COLS).sort("t")
    gt = folds = None
    if split == "train":
        gt = load(cfg, split, "gt").with_columns(label=pl.lit(1, pl.UInt8))
        folds = load(cfg, split, "folds", ["s1", "fold"]).sort("s1")
    out_dir = cfg.split_dir(split) / f"feat_{variant}"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("part_*.parquet"):
        old.unlink()
    parts = sorted((cfg.split_dir(split) / CAND_DIR[variant]).glob("part_*.parquet"))
    extended = variant == "c"
    comp = competition(cfg, split) if extended else None
    s_scale = None
    import countscale
    if extended and countscale.enabled():
        f = countscale.factors(s1, tg)
        s_scale = s1.join(f, on="country", how="left").sort("s1")["s_scale"].to_numpy()
    if extended:
        s1 = s1.join(load(cfg, split, "s1_sk"), on="s1", how="left").sort("s1")
        tg = tg.join(load(cfg, split, "tgt_sk"), on="t", how="left").sort("t")
    for i, part in enumerate(parts):
        cand = pl.read_parquet(part)
        if "rrank" in cand.columns:
            cand = cand.filter(pl.col("rrank") <= cfg.rk_cand_k)
        if comp is not None:
            cand = cand.join(comp, on="t", how="left")
            if s_scale is not None:
                cand = cand.with_columns(t_n_s1=pl.col("t_n_s1") * pl.Series(s_scale[cand["s1"].to_numpy()]))
        cand = cand.with_columns(
            cheap_rank=pl.col("cheap").rank("min", descending=True).over("s1"),
            cheap_rel=pl.col("cheap") / pl.col("cheap").max().over("s1"),
            log_n_cand=pl.len().over("s1").cast(pl.Float32).log1p(),
        )
        if gt is not None:
            cand = (
                cand.join(gt, on=["s1", "t"], how="left")
                .with_columns(pl.col("label").fill_null(0))
                .join(folds, on="s1", how="left")
            )
        outs = []
        for lo in range(0, cand.height, cfg.feat_chunk):
            outs.append(pair_features(cand.slice(lo, cfg.feat_chunk), s1, tg, extended))
        pl.concat(outs).write_parquet(out_dir / f"part_{i:03d}.parquet")
        log(f"[{split}] features part {i}: {cand.height:,} pairs ({time.time() - t0:.0f}s)")


def feature_parts(cfg: Config, split: str, variant: str = "a") -> list:
    return sorted((cfg.split_dir(split) / f"feat_{variant}").glob("part_*.parquet"))
