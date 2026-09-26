import os, sys, resource
os.environ["POLARS_MAX_THREADS"]="1"; os.environ["OMP_NUM_THREADS"]="1"
import polars as pl, numpy as np
sys.path.insert(0, "/home/user/Amazon_ml/v10/code_v10/code/business_entity_resolution/src")
from decide import prior_shift, single_owner, dta_select
from score import per_entity
D="/home/user/work_v10/train"; O="/home/user/Amazon_ml/alt/analysis"
folds = pl.scan_parquet(f"{D}/folds.parquet").filter((pl.col("fold")==4)&(pl.col("s1")<150000)).select("s1","stratum").collect()
samp = folds.filter((pl.col("s1").hash(seed=7) % 2) == 0)
samp = samp.sample(n=min(15000, samp.height), seed=1).sort("s1")
print("fold4 in range", folds.height, "sample", samp.height)
ids = samp["s1"].implode()
sc = pl.scan_parquet(f"{D}/scores_c.parquet").filter(pl.col("s1").is_in(ids)).collect(engine="streaming")
print("sample pairs", sc.height)
tset = sc.filter(pl.col("p")>1e-4)["t"].unique()
comp = (pl.scan_parquet(f"{D}/scores_c.parquet").select("s1","t","p")
        .filter(pl.col("t").is_in(tset.implode()) & (pl.col("p")>1e-4)).collect(engine="streaming"))
print("competing rows (all S1s on sample targets)", comp.height, "maxrss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
s = prior_shift(comp, 0.6)
s = single_owner(s)
s = s.filter(pl.col("s1").is_in(ids))
pred = dta_select(s.filter(pl.col("p")>1e-4), 0.0).select("s1","t")
gt = pl.scan_parquet(f"{D}/gt.parquet").filter(pl.col("s1").is_in(ids)).collect()
pe = per_entity(samp, pred, gt)
print("sample macro F0.5", pe["f"].mean())
# also within-sample-only single owner
s2 = single_owner(prior_shift(sc.select("s1","t","p"),0.6))
pred2 = dta_select(s2.filter(pl.col("p")>1e-4),0.0).select("s1","t")
print("within-sample single_owner macro F0.5", per_entity(samp,pred2,gt)["f"].mean())
sc = sc.join(s.select("s1","t",p_own="p"), on=["s1","t"], how="left").join(pred.with_columns(pred=pl.lit(1,pl.UInt8)), on=["s1","t"], how="left").with_columns(pl.col("pred").fill_null(0))
sc.write_parquet(f"{O}/sample_pairs.parquet"); gt.write_parquet(f"{O}/sample_gt.parquet"); pe.join(samp,on="s1").write_parquet(f"{O}/sample_entity.parquet")
print("maxrss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
