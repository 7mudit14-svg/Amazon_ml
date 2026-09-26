import os
os.environ["POLARS_MAX_THREADS"]="1"
import polars as pl
O="/home/user/Amazon_ml/alt/analysis"
fd=pl.read_parquet(f"{O}/sample_feat.parquet").join(pl.read_parquet(f"{O}/sample_seg.parquet"),on=["s1","t"])
tg=pl.read_parquet(f"{O}/sample_tgt.parquet").select("t","src","script",tn="business_name",ta="business_address")
s1=pl.read_parquet(f"{O}/sample_s1.parquet").select("s1",sn="business_name",sa="business_address")
d=fd.select("s1","t","label","pred","p","p1","seg").join(tg,on="t").join(s1,on="s1")
n=pl.col("tn"); a=pl.col("ta")
F={
 "t_accent": n.str.contains(r"[À-ÖØ-öø-ɏ]") & ~pl.col("sn").str.contains(r"[À-ÖØ-öø-ɏ]"),
 "t_digit_in_word": n.str.contains(r"[A-Za-z][0-9][A-Za-z]|\b[05][a-z]{2,}|\b[a-z]{2,}[0][a-z]"),
 "t_l_for_I": n.str.contains(r"\bl[a-z]") & ~pl.col("sn").str.contains(r"\bl[a-z]") ,
 "t_dup_word": n.str.to_lowercase().str.extract_all(r"[a-z0-9]{3,}").list.eval(pl.element().is_duplicated().any()).list.first().fill_null(False),
 "t_brackets": n.str.contains(r"[\[\(]") & ~pl.col("sn").str.contains(r"[\[\(]"),
 "t_idtag": n.str.contains(r"#\d+|\(ID: ?\d+\)"),
 "t_honorific": n.str.contains(r"(?i)^(dr|smt|sri|shri|mr|mrs|ms|m/s|the)\.?\s"),
 "t_double_space": n.str.contains(r"  "),
 "t_all_lower": (n==n.str.to_lowercase()) & n.str.contains(r"[a-z]"),
 "t_all_upper": (n==n.str.to_uppercase()) & n.str.contains(r"[A-Z]"),
 "exact_raw_name": n==pl.col("sn"),
 "t_addr_pobox": a.str.contains(r"(?i)\b(pmb|p\.?o\.? box)\b"),
 "t_addr_half": a.str.contains(r"\b1/2\b"),
 "t_addr_hashhash": a.str.contains(r"##"),
 "t_addr_null": a.str.contains(r"(?i)\bnull\b|<null>|n/a"),
}
d=d.with_columns(**F)
d.select(["s1","t"]+list(F)).write_parquet(f"{O}/feat_noise.parquet")
u=d.filter((pl.col("p")>0.02)&(pl.col("p")<0.98))
rows=[]
for f in F:
    for zone,dd in [("all",d),("unc",u)]:
        on=dd.filter(pl.col(f)); off=dd.filter(~pl.col(f))
        rows.append(dict(feat=f,zone=zone,n_on=on.height,true_on=on["label"].mean(),p_on=on["p"].mean(),n_off=off.height,true_off=off["label"].mean(),p_off=off["p"].mean(),
          fn_on=on.filter((pl.col("label")==1)&(pl.col("pred")==0)).height, fp_on=on.filter((pl.col("label")==0)&(pl.col("pred")==1)).height))
pl.Config.set_tbl_rows(50); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)
print(pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(3)))
