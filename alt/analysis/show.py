import os,sys
os.environ["POLARS_MAX_THREADS"]="1"
import polars as pl
O="/home/user/Amazon_ml/alt/analysis"
cat=sys.argv[1]; n=int(sys.argv[2]); seed=int(sys.argv[3]) if len(sys.argv)>3 else 0
err=pl.read_parquet(f"{O}/errors.parquet").filter(pl.col("cat")==cat)
s1=pl.read_parquet(f"{O}/sample_s1.parquet"); tg=pl.read_parquet(f"{O}/sample_tgt.parquet"); fd=pl.read_parquet(f"{O}/sample_feat.parquet")
e=err.sample(n=min(n,err.height),seed=seed).join(s1.select("s1","business_name","business_address","country","core"),on="s1").join(tg.select("t","src",tn="business_name",ta="business_address",tcore="core",tscript="script"),on="t").join(fd.drop("label","pred","p","p_own"),on=["s1","t"],how="left")
for r in e.iter_rows(named=True):
    print(f"[{r['s1']}/{r['t']} {r['country']} src{r['src']} p={r['p'] and round(r['p'],3)} pown={r['p_own'] and round(r['p_own'],3)} ntrue={r['n_true']} npred={r['n_pred']} tp={r['tp']}]")
    print(f"   S: {r['business_name']!r} | {r['business_address']!r}")
    print(f"   T: {r['tn']!r} | {r['ta']!r}")
    if r['p1'] is not None:
        print(f"   p1={r['p1']:.3f} tset={r['name_tset']:.2f} ratio={r['name_ratio']:.2f} atset={r['addr_tset']:.2f} nm={r['num_match']:.0f} nc={r['num_conflict']:.0f} near={r['num_near']:.0f} tbl={r['t_blank']:.0f} sbl={r['s_blank']:.0f} ind={r['t_indic']:.0f} dom={r['t_domain']:.0f} legc={r['legal_conflict']:.0f} cmax={r['c_other_max']:.2f} rank={r['s_rank']:.0f} dup={r['sib_dup']:.0f} tn={r['t_n_s1']:.2f} nwnew={r['nw_n_new']:.0f}")
