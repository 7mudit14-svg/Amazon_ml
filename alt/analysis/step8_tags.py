import os, sys
os.environ["POLARS_MAX_THREADS"]="1"; os.environ["OMP_NUM_THREADS"]="1"
import polars as pl, numpy as np
from rapidfuzz import process, fuzz
sys.path.insert(0, "/home/user/Amazon_ml/v10/code_v10/code/business_entity_resolution/src")
from decide import prior_shift, single_owner, dta_select
from score import per_entity
O="/home/user/Amazon_ml/alt/analysis"
pe=pl.read_parquet(f"{O}/sample_entity.parquet"); gt=pl.read_parquet(f"{O}/sample_gt.parquet")
sp=pl.read_parquet(f"{O}/sample_pairs.parquet")
for r in (1.0,0.8,0.6):
    s=single_owner(prior_shift(sp.select("s1","t","p"),r)); pred=dta_select(s.filter(pl.col("p")>1e-4),0.0)
    print("shift",r,"macro",per_entity(pe.select("s1"),pred,gt)["f"].mean())
err=pl.read_parquet(f"{O}/errors.parquet")
s1=pl.read_parquet(f"{O}/sample_s1.parquet"); tg=pl.read_parquet(f"{O}/sample_tgt.parquet")
fb=pl.read_parquet(f"{O}/feat_blank.parquet").select("s1","t","rr_gap","n_claim")
e=err.join(s1.select("s1",sn="business_name",sa="business_address",score="core",slegal="legal",snums="addr_nums",satok="addr_toks",sblank="addr_blank",skey="core_key"),on="s1") \
     .join(tg.select("t","src",tn="business_name",ta="business_address",tcore="core",tlegal="legal",tnums="addr_nums",tatok="addr_toks",tblank="addr_blank",tdom="is_domain",tscript="script",tkey="core_key"),on="t") \
     .join(fb,on=["s1","t"],how="left")
e=e.with_columns(ntset=pl.Series(process.cpdist(e["score"].to_list(),e["tcore"].to_list(),scorer=fuzz.token_set_ratio,workers=1)/100.0),
                 atset=pl.Series(process.cpdist(e["satok"].to_list(),e["tatok"].to_list(),scorer=fuzz.token_set_ratio,workers=1)/100.0))
sn0=pl.col("snums").str.split(" ").list.first(); tn0=pl.col("tnums").str.split(" ").list.first()
unit=r"(?i)\b(unit|suite|ste|apt|fl|floor|flat|#)\b|#"
tags={
 "indic_script": pl.col("tscript").is_in(["indic","mixed"]),
 "website_name": pl.col("tdom") | pl.col("tn").str.contains(r"(?i)\.com|www\.|\.in\b|\.net|\.org"),
 "dba_fka": pl.col("tn").str.contains(r"(?i)\b(d/?b/?a|dba|fka|f/k/a|aka|formerly|trading as|t/a)\b") | pl.col("sn").str.contains(r"(?i)\b(d/?b/?a|dba|fka|aka|formerly|trading as)\b"),
 "alias_name": (pl.col("ntset")<0.5),
 "blank_t_addr": pl.col("tblank"),
 "blank_tie_identical_s1names": pl.col("tblank") & (pl.col("rr_gap").abs()<0.001) & (pl.col("n_claim")>=2),
 "chain_common_name": pl.lit(False),
 "word_order": (pl.col("tkey")!=pl.col("skey")) & (pl.col("tcore").str.split(" ").list.sort()==pl.col("score").str.split(" ").list.sort()),
 "typo_name": (pl.col("ntset")>=0.5)&(pl.col("ntset")<0.9),
 "legal_variant": (pl.col("slegal")!=pl.col("tlegal")),
 "num_range": pl.col("ta").str.contains(r"\b\d+\s?-\s?\d+\b") | pl.col("sa").str.contains(r"\b\d+\s?-\s?\d+\b"),
 "missing_house_no": (pl.col("snums")!="") & (pl.col("tnums")=="") & ~pl.col("tblank"),
 "house_no_diff": (pl.col("snums")!="") & (pl.col("tnums")!="") & (sn0!=tn0),
 "house_no_trunc": (pl.col("snums")!="") & (pl.col("tnums")!="") & (sn0!=tn0) & (sn0.str.contains(tn0,literal=True)|tn0.str.contains(sn0,literal=True)),
 "unit_diff": pl.col("sa").str.contains(unit) != pl.col("ta").str.contains(unit),
 "pobox_pmb_half": pl.col("ta").str.contains(r"(?i)\b(pmb|p\.?o\.? box)\b|\b1/2\b"),
 "addr_weak": (pl.col("atset")<0.6)&~pl.col("tblank"),
}
e=e.with_columns(**{k:v.fill_null(False) for k,v in tags.items()})
tot=(1-pe["f"]).mean()
rows=[]
for cat in ["FN_cand","FP","FN_block"]:
    x=e.filter(pl.col("cat")==cat)
    for k in tags:
        y=x.filter(pl.col(k))
        rows.append(dict(cat=cat,tag=k,n=y.height,share_of_cat=round(y.height/max(x.height,1),3),loss_share=round(y["loss"].sum()/tot,4)))
pl.Config.set_tbl_rows(80)
print(pl.DataFrame(rows).filter(pl.col("n")>0).sort(["cat","loss_share"],descending=[False,True]))
# primary (exclusive) assignment in priority order
prio=["indic_script","website_name","dba_fka","alias_name","blank_tie_identical_s1names","blank_t_addr","word_order","typo_name","house_no_trunc","house_no_diff","missing_house_no","legal_variant","unit_diff","pobox_pmb_half","addr_weak"]
expr=pl.lit("other")
for k in reversed(prio): expr=pl.when(pl.col(k)).then(pl.lit(k)).otherwise(expr)
e=e.with_columns(primary=expr)
print(e.group_by("cat","primary").agg(n=pl.len(),loss_share=(pl.col("loss").sum()/tot).round(4)).sort("loss_share",descending=True))
e.write_parquet(f"{O}/errors_tagged.parquet")
