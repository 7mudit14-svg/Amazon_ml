import os
os.environ["POLARS_MAX_THREADS"]="1"
import polars as pl
D="/home/user/work_v10/train"; O="/home/user/Amazon_ml/alt/analysis"
sp = pl.read_parquet(f"{O}/sample_pairs.parquet"); gt = pl.read_parquet(f"{O}/sample_gt.parquet"); pe = pl.read_parquet(f"{O}/sample_entity.parquet")
ids = pe["s1"].implode()
N = pe.height
miss = gt.join(sp.select("s1","t"), on=["s1","t"], how="anti").with_columns(cat=pl.lit("FN_block"), label=pl.lit(1,pl.UInt8), pred=pl.lit(0,pl.UInt8))
err = pl.concat([
  sp.filter((pl.col("label")==1)&(pl.col("pred")==0)).select("s1","t","label","pred","p","p_own").with_columns(cat=pl.lit("FN_cand")),
  sp.filter((pl.col("label")==0)&(pl.col("pred")==1)).select("s1","t","label","pred","p","p_own").with_columns(cat=pl.lit("FP")),
  miss.with_columns(p=pl.lit(None,pl.Float64), p_own=pl.lit(None,pl.Float64)).select("s1","t","label","pred","p","p_own","cat"),
])
err = err.with_columns(n_err=pl.len().over("s1")).join(pe.select("s1","f","n_true","n_pred","tp","stratum"), on="s1").with_columns(loss=(1-pl.col("f"))/pl.col("n_err")/N)
print("total macro loss", (1-pe["f"]).mean())
print(err.group_by("cat").agg(n=pl.len(), loss=pl.col("loss").sum(), share=pl.col("loss").sum()/(1-pe["f"]).mean()).sort("loss",descending=True))
# attach features
cols = ["s1","t","p1","name_tset","name_ratio","name_pratio","addr_tset","addr_jac","num_match","num_conflict","num_near","num_suffix","t_blank","s_blank","t_indic","t_domain","legal_conflict","leg_swap","c_other_max","s_rank","sib_dup","src3","t_n_s1","sk_tset","num_off","nw_n_new","nw_n_gone","nw_new_df_min"]
fd = pl.concat([pl.scan_parquet(f"{D}/feat_d/part_{i:03d}.parquet").select(cols).filter(pl.col("s1").is_in(ids)).collect() for i in range(5)])
print("feat rows", fd.height, "sample pairs", sp.height)
fd.join(sp.select("s1","t","label","pred","p","p_own"), on=["s1","t"]).write_parquet(f"{O}/sample_feat.parquet")
tt = pl.concat([sp["t"], miss["t"]]).unique().implode()
F = ["business_name","business_address","country","core","core_key","legal","concat","addr_nums","addr_toks","addr_blank","is_domain","script"]
s1 = pl.scan_parquet(f"{D}/s1.parquet").select(["s1"]+F).filter(pl.col("s1").is_in(ids)).collect()
tg = pl.scan_parquet(f"{D}/tgt.parquet").select(["t","src"]+F).filter(pl.col("t").is_in(tt)).collect(engine="streaming")
s1.write_parquet(f"{O}/sample_s1.parquet"); tg.write_parquet(f"{O}/sample_tgt.parquet")
err.write_parquet(f"{O}/errors.parquet")
import resource; print("maxrss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
