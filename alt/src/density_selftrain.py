"""Step 2: does test-time self-training help at test density? Measured on the US stress-v2 split.

Stage 2 is refit on labelled train dev rows (dom=0) + pseudo-labels from the stress split's own
v10 stage-2 scores (dom=1, weight 2; pos p>=0.98 & num_match, neg p<=0.02 sampled 10%). Stress
pairs of fold-4 visible S1s are rescored with dom=1 and scored against their real labels (labels
never used above). Baseline = v10 scores on the same split with the same decision rule.
Features are streamed part by part to stay inside 15 GB.
"""
import glob, sys, time
import numpy as np, polars as pl, lightgbm as lgb
from v10env import WORK
from stack import FEATURES_D, PARAMS
from decide import dta_select, prior_shift, single_owner
from score import per_entity

t0 = time.time(); T = f"{WORK}/train"; S = sys.argv[1] if len(sys.argv) > 1 else "/home/user/work_s2"
F2 = list(FEATURES_D) + ["dom"]
keep_neg = (pl.struct("s1", "t").hash(seed=5) % 100) < 10
parts = sorted(glob.glob(f"{T}/feat_d/part_*.parquet"))
tr = pl.concat([pl.read_parquet(p).filter((pl.col("fold") != 4) & ((pl.col("label") == 1) | keep_neg)) for p in parts[:15]]).with_columns(dom=pl.lit(0.0, pl.Float32))
va = pl.concat([pl.read_parquet(p).filter(pl.col("fold") == 4) for p in parts[35:37]]).sample(fraction=0.3, seed=1).with_columns(dom=pl.lit(0.0, pl.Float32))
gt = pl.read_parquet(f"{T}/gt.parquet"); folds = pl.read_parquet(f"{T}/folds.parquet", columns=["s1", "fold"])
ms1 = pl.read_parquet(f"{S}/map_s1.parquet"); mt = pl.read_parquet(f"{S}/map_t.parquet")
ev_new = ms1.join(folds, left_on="orig_s1", right_on="s1").filter(pl.col("fold") == 4).select("s1")
ev = ms1.join(ev_new, on="s1").select(s1="orig_s1")
p2 = pl.read_parquet(f"{S}/test/scores_c.parquet", columns=["s1", "t", "p"])
pos_l, neg_l, ev_l = [], [], []
for f in sorted(glob.glob(f"{S}/test/feat_d/part_*.parquet")):
    x = pl.read_parquet(f).join(p2, on=["s1", "t"]).with_columns(dom=pl.lit(1.0, pl.Float32)).select(["s1", "t", "p"] + F2)
    pos_l.append(x.filter((pl.col("p") >= 0.98) & (pl.col("num_match") == 1)))
    neg_l.append(x.filter((pl.col("p") <= 0.02) & keep_neg))
    ev_l.append(x.join(ev_new, on="s1", how="semi"))
ps = pl.concat([pl.concat(pos_l).with_columns(label=pl.lit(1, pl.UInt8)), pl.concat(neg_l).with_columns(label=pl.lit(0, pl.UInt8))])
st = pl.concat(ev_l); del pos_l, neg_l, ev_l
print(f"train {tr.height:,}, pseudo {ps.height:,}, eval pairs {st.height:,} ({time.time()-t0:.0f}s)", flush=True)
def refit(with_pseudo):
    X = pl.concat([tr.select(F2 + ["label"]), ps.select(F2 + ["label"])]) if with_pseudo else tr.select(F2 + ["label"])
    wt = np.where(X["label"].to_numpy() == 1, 1.0, 10.0) * (np.r_[np.ones(tr.height), np.full(ps.height, 2.0)] if with_pseudo else 1.0)
    d = lgb.Dataset(X.select(F2).to_numpy(), X["label"].to_numpy(), weight=wt, feature_name=F2)
    dv = lgb.Dataset(va.select(F2).to_numpy(), va["label"].to_numpy(), weight=np.where(va["label"].to_numpy() == 1, 1.0, 10.0), reference=d)
    b = lgb.train(PARAMS, d, num_boost_round=800, valid_sets=[dv], callbacks=[lgb.early_stopping(30, verbose=False)])
    return st.select("s1", "t").with_columns(p=pl.Series(b.predict(st.select(F2).to_numpy())))
fresh = refit(False)
new = refit(True)


def f05(sc, shift):
    # single_owner over the eval S1s' claims only (same for both arms)
    pred = dta_select(single_owner(prior_shift(sc, shift)).filter(pl.col("p") > 1e-4), 0.0).select("s1", "t")
    pred = pred.join(ms1, on="s1").join(mt, on="t").select(s1="orig_s1", t="orig_t")
    return per_entity(ev, pred, gt)["f"].mean()


p2e = p2.join(ev_new, on="s1", how="semi")
for s in (1.0, 0.8, 0.6, 0.45):
    print(f"shift {s}: v10 {f05(p2e, s):.5f} | fresh no-pseudo {f05(fresh, s):.5f} | fresh + pseudo {f05(new, s):.5f} ({time.time()-t0:.0f}s)", flush=True)
