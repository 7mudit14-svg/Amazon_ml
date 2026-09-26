import os
os.environ["POLARS_MAX_THREADS"]="1"
import polars as pl
D="/home/user/work_v10/train"; O="/home/user/Amazon_ml/alt/analysis"
fd=pl.read_parquet(f"{O}/sample_feat.parquet").join(pl.read_parquet(f"{O}/sample_seg.parquet"),on=["s1","t"])
A=fd.filter(pl.col("seg")=="A_blank_name")
print(A.group_by("label").agg(pl.col("p1","c_other_max","t_n_s1","s_rank","name_ratio","sib_dup").mean(), n=pl.len()))
print("label rate by name_ratio==1:", A.group_by(pl.col("name_ratio")>=0.999).agg(pl.col("label").mean(), pl.len()))
# competitors for some FN targets
err=pl.read_parquet(f"{O}/errors.parquet").filter(pl.col("cat")=="FN_cand").join(A.select("s1","t"),on=["s1","t"],how="semi").sample(8,seed=3)
ts=err["t"].implode()
comp=pl.scan_parquet(f"{D}/scores_c.parquet").filter(pl.col("t").is_in(ts)).collect(engine="streaming")
s1=pl.scan_parquet(f"{D}/s1.parquet").select("s1","business_name","business_address").filter(pl.col("s1").is_in(comp["s1"].implode())).collect()
tg=pl.read_parquet(f"{O}/sample_tgt.parquet")
for r in err.iter_rows(named=True):
    print("=== T", tg.filter(pl.col("t")==r["t"]).select("business_name","src").row(0), "owner", r["s1"])
    c=comp.filter(pl.col("t")==r["t"]).join(s1,on="s1").sort("p",descending=True)
    for x in c.head(8).iter_rows(named=True): print(f"   {x['s1']} lab={x['label']} p={x['p']:.3f} {x['business_name']!r} | {x['business_address'][:60]!r}")
    print("   n claimants", c.height)
    # owner's other targets
    own=pl.read_parquet(f"{O}/sample_pairs.parquet").filter((pl.col("s1")==r["s1"])&((pl.col("label")==1)|(pl.col("pred")==1))).join(tg.select("t","business_name","business_address","src"),on="t")
    for x in own.iter_rows(named=True): print(f"      own lab={x['label']} pred={x['pred']} p={x['p']:.3f} src{x['src']} {x['business_name']!r} | {x['business_address'][:50]!r}")
