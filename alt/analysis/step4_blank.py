import os, resource
os.environ["POLARS_MAX_THREADS"]="1"
import polars as pl, numpy as np
from rapidfuzz import process, fuzz
from rapidfuzz.utils import default_process
import unicodedata
D="/home/user/work_v10/train"; O="/home/user/Amazon_ml/alt/analysis"
fd=pl.read_parquet(f"{O}/sample_feat.parquet").join(pl.read_parquet(f"{O}/sample_seg.parquet"),on=["s1","t"])
tg=pl.read_parquet(f"{O}/sample_tgt.parquet")
gt=pl.read_parquet(f"{O}/sample_gt.parquet")
# structural prior: blank true targets per S1 per source
g=gt.join(tg.select("t","src","addr_blank"),on="t").group_by("s1","src").agg(nb=pl.col("addr_blank").sum())
print("blank true targets per (S1,src):", g.group_by("nb").len().sort("nb"))
A_t = fd.filter(pl.col("seg").is_in(["A_blank_name","A2_blank_weakname"]))["t"].unique()
comp = pl.scan_parquet(f"{D}/scores_c.parquet").select("s1","t","p").filter(pl.col("t").is_in(A_t.implode())).collect(engine="streaming")
cs1 = comp["s1"].unique()
print("A targets", A_t.len(), "comp rows", comp.height, "comp S1s", cs1.len())
s1n = pl.scan_parquet(f"{D}/s1.parquet").select("s1","business_name").filter(pl.col("s1").is_in(cs1.implode())).collect()
def norm(x):
    x=unicodedata.normalize("NFKD",x); x="".join(c for c in x if not unicodedata.combining(c)); return default_process(x)
s1n=s1n.with_columns(sn=pl.col("business_name").map_elements(norm,return_dtype=pl.String))
tn=tg.filter(pl.col("t").is_in(A_t.implode())).select("t",tn=pl.col("business_name").map_elements(norm,return_dtype=pl.String))
comp=comp.join(s1n.select("s1","sn"),on="s1").join(tn,on="t")
comp=comp.with_columns(rr=pl.Series(process.cpdist(comp["sn"].to_list(),comp["tn"].to_list(),scorer=fuzz.ratio,workers=1)/100.0),
                       rt=pl.Series(process.cpdist(comp["sn"].to_list(),comp["tn"].to_list(),scorer=fuzz.token_sort_ratio,workers=1)/100.0))
comp=comp.with_columns(
    rr_other_max=(pl.col("rr").max().over("t") if False else pl.lit(0.0)))
# max over others: use top2 trick
def other_max(col):
    mx=pl.col(col).max().over("t"); n_at=(pl.col(col)==mx).sum().over("t")
    sec=pl.col(col).filter(pl.col(col)<mx).max().over("t")
    return pl.when((pl.col(col)<mx)|(n_at>1)).then(mx).otherwise(sec.fill_null(0.0))
comp=comp.with_columns(rr_om=other_max("rr"), rt_om=other_max("rt"), n_claim=pl.len().over("t"))
comp=comp.with_columns(rr_gap=pl.col("rr")-pl.col("rr_om"), rt_gap=pl.col("rt")-pl.col("rt_om"))
# competitor "blank slot" status: does each claimant S1 already hold a confident blank target in same src?
conf = pl.scan_parquet(f"{D}/scores_c.parquet").select("s1","t","p").filter((pl.col("p")>0.5)&pl.col("s1").is_in(cs1.implode())).collect(engine="streaming")
tinfo = pl.scan_parquet(f"{D}/tgt.parquet").select("t","src","addr_blank").filter(pl.col("t").is_in(pl.concat([conf["t"],A_t]).unique().implode())).collect(engine="streaming")
conf=conf.join(tinfo,on="t")
tsrc=tinfo.filter(pl.col("t").is_in(A_t.implode())).select("t",tsrc="src")
comp=comp.join(tsrc,on="t")
cb=conf.filter(pl.col("addr_blank")).select("s1",cb_t="t",cb_src="src",cb_p="p")
x=comp.join(cb,left_on=["s1","tsrc"],right_on=["s1","cb_src"],how="left").filter(pl.col("cb_t").is_null()|(pl.col("cb_t")!=pl.col("t")))
x=x.group_by("s1","t").agg(has_blank_same=pl.col("cb_t").is_not_null().any())
comp=comp.join(x,on=["s1","t"],how="left").with_columns(pl.col("has_blank_same").fill_null(False))
# any confident same-src (any address) count
cs=conf.join(tsrc.rename({"t":"tt"}),left_on="t",right_on="tt",how="anti").group_by("s1","src").agg(nconf_src=pl.len())
comp=comp.join(cs,left_on=["s1","tsrc"],right_on=["s1","src"],how="left").with_columns(pl.col("nconf_src").fill_null(0))
# competitor with best p (other than me): its has_blank_same
def at_best_other(col):
    pass
comp=comp.sort(["t","p"],descending=[False,True]).with_columns(r=pl.int_range(pl.len()).over("t"))
top=comp.filter(pl.col("r")<=1).select("t","r","s1","has_blank_same","p")
t0=top.filter(pl.col("r")==0).select("t",b0s1="s1",b0hb="has_blank_same"); t1=top.filter(pl.col("r")==1).select("t",b1hb="has_blank_same")
comp=comp.join(t0,on="t").join(t1,on="t",how="left").with_columns(
   comp_best_has_blank=pl.when(pl.col("s1")==pl.col("b0s1")).then(pl.col("b1hb")).otherwise(pl.col("b0hb")).fill_null(False))
res=fd.select("s1","t","label","pred","p","p1","seg","name_tset","c_other_max").join(comp.select("s1","t","rr","rt","rr_gap","rt_gap","n_claim","has_blank_same","comp_best_has_blank","nconf_src"),on=["s1","t"],how="inner")
res.write_parquet(f"{O}/feat_blank.parquet")
print("maxrss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
