"""Step 1: reproduce v10's sealed-fold score from its stored out-of-fold stage-2 scores,
then split the loss by country x error type."""
import json
import polars as pl
from v10env import WORK, s1_train, tgt_train
from matcher_b import apply_policy
from score import per_entity

meta = json.load(open(f"{WORK}/models/matcher_c/meta.json"))
sc = pl.read_parquet(f"{WORK}/train/scores_c.parquet", columns=["s1", "t", "p"])
gt = pl.read_parquet(f"{WORK}/train/gt.parquet")
folds = pl.read_parquet(f"{WORK}/train/folds.parquet", columns=["s1", "fold"])
s1 = s1_train().join(folds, on="s1")
sealed = s1.filter(pl.col("fold") == 4)
pool = sc.filter(pl.col("p") > 1e-4)
pred = apply_policy(pool, meta["policy"])
ent = per_entity(sealed.select("s1"), pred, gt).join(sealed, on="s1")
print("policy", meta["policy"], "sealed macro F0.5 =", round(ent["f"].mean(), 5),
      "| meta says", round(meta["sealed"]["overall"]["macro_f05"], 5))
print(ent.group_by("country").agg(n=pl.len(), f=pl.col("f").mean()))

# Loss decomposition on the sealed fold (weights = contribution to macro loss).
tg = tgt_train().select("t", ta="business_address")
blank = pl.col("ta").is_null() | pl.col("ta").str.to_lowercase().is_in(["none", "null", "n/a", ""])
gts = gt.join(sealed.select("s1"), on="s1", how="semi").join(tg, on="t").with_columns(t_blank=blank)
cand = sc.join(sealed.select("s1"), on="s1", how="semi").select("s1", "t")
gts = gts.join(cand.with_columns(inc=pl.lit(True)), on=["s1", "t"], how="left").join(
    pred.select("s1", "t", hit=pl.lit(True)), on=["s1", "t"], how="left").with_columns(
    pl.col("inc", "hit").fill_null(False))
fn = gts.filter(~pl.col("hit")).with_columns(
    kind=pl.when(~pl.col("inc")).then(pl.lit("not_retrieved"))
    .when(pl.col("t_blank")).then(pl.lit("rejected_blank")).otherwise(pl.lit("rejected_addr")))
fp = pred.join(sealed.select("s1"), on="s1", how="semi").join(gt, on=["s1", "t"], how="anti")
# Attribute each S1's loss (1-f) to its error events proportionally.
ev = pl.concat([fn.select("s1", "kind"), fp.select("s1", kind=pl.lit("false_match"))])
ev = ev.join(ev.group_by("s1").agg(k=pl.len()), on="s1").join(ent.select("s1", "f", "country"), on="s1")
tab = (ev.group_by("country", "kind").agg(loss=((1 - pl.col("f")) / pl.col("k")).sum() / 1.0)
       .join(sealed.group_by("country").agg(n=pl.len()), on="country")
       .with_columns(loss_share_of_macro=pl.col("loss") / pl.col("n")).sort("country", "kind"))
print(tab.select("country", "kind", "loss_share_of_macro"))
print("pair recall in candidates:", gts["inc"].mean(), " blank-target share:", gts["t_blank"].mean())
