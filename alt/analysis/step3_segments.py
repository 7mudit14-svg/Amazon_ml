import os
os.environ["POLARS_MAX_THREADS"]="1"
import polars as pl
O="/home/user/Amazon_ml/alt/analysis"
fd=pl.read_parquet(f"{O}/sample_feat.parquet"); err=pl.read_parquet(f"{O}/errors.parquet")
seg = (pl.when(pl.col("t_indic")==1).then(pl.lit("D_indic"))
 .when(pl.col("t_domain")==1).then(pl.lit("C_domain"))
 .when((pl.col("t_blank")==1)&(pl.col("name_tset")>=0.9)).then(pl.lit("A_blank_name"))
 .when((pl.col("t_blank")==1)).then(pl.lit("A2_blank_weakname"))
 .when((pl.col("addr_tset")>=0.8)&(pl.col("name_tset")<0.5)).then(pl.lit("B_alias_addr"))
 .when((pl.col("name_tset")>=0.9)&((pl.col("num_conflict")==1)|(pl.col("num_match")==0))).then(pl.lit("E_name_numdiff"))
 .when((pl.col("name_tset")>=0.9)&(pl.col("addr_tset")<0.6)).then(pl.lit("G_name_addrdiff"))
 .when(pl.col("name_tset")>=0.9).then(pl.lit("H_name_addr_ok"))
 .otherwise(pl.lit("F_partial_name")))
fd=fd.with_columns(seg=seg)
fd.select("s1","t","seg").write_parquet(f"{O}/sample_seg.parquet")
e=err.join(fd.select("s1","t","seg"),on=["s1","t"],how="left").with_columns(pl.col("seg").fill_null("Z_blockmiss"))
tot=err["loss"].sum()
print(e.group_by("cat","seg").agg(n=pl.len(),loss_share=pl.col("loss").sum()/tot).sort("loss_share",descending=True))
# candidate-level base rates per segment in 'uncertain' zone
print(fd.group_by("seg").agg(n=pl.len(),pos=pl.col("label").sum(),pred=pl.col("pred").sum(),
   unc=((pl.col("p")>0.02)&(pl.col("p")<0.98)).sum(), unc_pos=(((pl.col("p")>0.02)&(pl.col("p")<0.98))&(pl.col("label")==1)).sum()).sort("seg"))
