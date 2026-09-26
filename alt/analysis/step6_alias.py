import os, resource
os.environ["POLARS_MAX_THREADS"]="1"
import polars as pl
D="/home/user/work_v10/train"; O="/home/user/Amazon_ml/alt/analysis"
fd=pl.read_parquet(f"{O}/sample_feat.parquet").select("s1","t","p1")
tg=pl.read_parquet(f"{O}/sample_tgt.parquet").select("t","core_key","country","addr_toks","addr_nums","is_domain","script")
d=fd.join(tg,on="t")
keys=d["core_key"].unique()
df=(pl.scan_parquet(f"{D}/tgt.parquet").select("core_key","country","addr_toks")
    .filter(pl.col("core_key").is_in(keys.implode()))
    .group_by("core_key","country").agg(t_core_df=pl.len(), t_core_naddr=pl.col("addr_toks").n_unique()).collect(engine="streaming"))
d=d.join(df,on=["core_key","country"],how="left")
# siblings sharing target core_key within this S1's list (excluding self)
d=d.with_columns(
  sib_tcore_n=(pl.len().over("s1","core_key")-1),
  sib_tcore_maxp=pl.when(pl.len().over("s1","core_key")>1).then(
      pl.col("p1").sort(descending=True).over("s1","core_key", mapping_strategy="join").list.eval(pl.element()).list.first()).otherwise(0.0))
# simpler exact: max of others = if self is the max then second max else max
g=pl.col("p1"); mx=g.max().over("s1","core_key"); cnt_mx=(g==mx).sum().over("s1","core_key")
sec=g.filter(g<mx).max().over("s1","core_key")
d=d.with_columns(sib_tcore_maxp=pl.when(pl.len().over("s1","core_key")==1).then(0.0).when((g<mx)|(cnt_mx>1)).then(mx).otherwise(sec.fill_null(0.0)))
d.select("s1","t","t_core_df","t_core_naddr","sib_tcore_n","sib_tcore_maxp").write_parquet(f"{O}/feat_alias.parquet")
print(d.select("t_core_df","sib_tcore_n","sib_tcore_maxp").describe())
print("maxrss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
