import os,sys
os.environ["POLARS_MAX_THREADS"]="1"
import polars as pl
O="/home/user/Amazon_ml/alt/analysis"
cat,seg,n=sys.argv[1],sys.argv[2],int(sys.argv[3])
err=pl.read_parquet(f"{O}/errors.parquet").filter(pl.col("cat")==cat).join(pl.read_parquet(f"{O}/sample_seg.parquet").filter(pl.col("seg")==seg),on=["s1","t"])
s1=pl.read_parquet(f"{O}/sample_s1.parquet"); tg=pl.read_parquet(f"{O}/sample_tgt.parquet"); fd=pl.read_parquet(f"{O}/sample_feat.parquet")
e=err.sample(n=min(n,err.height),seed=5).join(s1.select("s1","business_name","business_address","addr_nums"),on="s1").join(tg.select("t","src",tn="business_name",ta="business_address",tnums="addr_nums"),on="t").join(fd.drop("label","pred","p","p_own"),on=["s1","t"],how="left")
for r in e.iter_rows(named=True):
    print(f"[{r['s1']}/{r['t']} src{r['src']} p={r['p']:.3f} p1={r['p1']:.3f} ntrue={r['n_true']} npred={r['n_pred']} tp={r['tp']} cmax={r['c_other_max']:.2f} nm={r['num_match']:.0f} nc={r['num_conflict']:.0f} near={r['num_near']:.0f} suf={r['num_suffix']:.0f} off={r['num_off']:.2f} atset={r['addr_tset']:.2f} ntset={r['name_tset']:.2f}]")
    print(f"   S: {r['business_name']!r} | {r['business_address']!r} nums={r['addr_nums']!r}")
    print(f"   T: {r['tn']!r} | {r['ta']!r} nums={r['tnums']!r}")
