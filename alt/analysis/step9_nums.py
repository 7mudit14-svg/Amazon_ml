import polars as pl, numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
fd=pl.read_parquet("sample_feat.parquet").select("s1","t","label","pred","p","p1","name_tset","num_off")
tg=pl.read_parquet("sample_tgt.parquet").select("t",tnums="addr_nums",ta="business_address")
s1=pl.read_parquet("sample_s1.parquet").select("s1",snums="addr_nums",sa="business_address")
d=fd.join(tg,on="t").join(s1,on="s1").filter((pl.col("name_tset")>=0.9)&(pl.col("snums")!="")&(pl.col("tnums")!=""))
d=d.with_columns(s0=pl.col("snums").str.split(" ").list.first(), t0=pl.col("tnums").str.split(" ").list.first(),
   sset=pl.col("snums").str.split(" "), tset=pl.col("tnums").str.split(" "))
d=d.filter(pl.col("s0")!=pl.col("t0"))
d=d.with_columns(lev=pl.Series(process.cpdist(d["s0"].to_list(),d["t0"].to_list(),scorer=Levenshtein.distance,workers=1)),
  absdiff=(pl.col("s0").cast(pl.Int64,strict=False)-pl.col("t0").cast(pl.Int64,strict=False)).abs(),
  samelen=pl.col("s0").str.len_chars()==pl.col("t0").str.len_chars(),
  t0_in_s=pl.col("t0").is_in(pl.col("sset")), s0_in_t=pl.col("s0").is_in(pl.col("tset")),
  lead0=pl.col("t0").str.strip_chars_start("0")==pl.col("s0").str.strip_chars_start("0"),
  prefix_trunc=pl.col("s0").str.starts_with(pl.col("t0"))|pl.col("t0").str.starts_with(pl.col("s0")),
  suffix_trunc=pl.col("s0").str.ends_with(pl.col("t0"))|pl.col("t0").str.ends_with(pl.col("s0")),
  first_digit_same=pl.col("s0").str.slice(0,1)==pl.col("t0").str.slice(0,1),
  last_digit_same=pl.col("s0").str.slice(-1,1)==pl.col("t0").str.slice(-1,1),
)
u=d.filter((pl.col("p")>0.02)&(pl.col("p")<0.98))
print("all first-num-diff name-match pairs",d.height,"true",d["label"].mean(),"uncertain",u.height,"true",u["label"].mean(),"meanp",u["p"].mean())
for f in ["t0_in_s","s0_in_t","lead0","prefix_trunc","suffix_trunc","samelen","first_digit_same","last_digit_same"]:
    print(f, u.group_by(f).agg(n=pl.len(),true=pl.col("label").mean().round(3),mp=pl.col("p").mean().round(3),fn=((pl.col("label")==1)&(pl.col("pred")==0)).sum(),fp=((pl.col("label")==0)&(pl.col("pred")==1)).sum()).sort(f).rows())
print("lev",u.group_by(pl.col("lev").clip(0,4)).agg(n=pl.len(),true=pl.col("label").mean().round(3),mp=pl.col("p").mean().round(3)).sort("lev").rows())
print("absdiff samelen",u.filter(pl.col("samelen")).group_by(pl.col("absdiff").cut([1,2,5,10,100,1000])).agg(n=pl.len(),true=pl.col("label").mean().round(3),mp=pl.col("p").mean().round(3)).sort("absdiff").rows())
