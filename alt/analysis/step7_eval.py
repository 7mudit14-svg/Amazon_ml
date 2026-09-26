import os, sys
os.environ["POLARS_MAX_THREADS"]="1"; os.environ["OMP_NUM_THREADS"]="1"
import polars as pl, numpy as np, lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score
sys.path.insert(0, "/home/user/Amazon_ml/v10/code_v10/code/business_entity_resolution/src")
from decide import prior_shift, single_owner, dta_select
from score import per_entity
O="/home/user/Amazon_ml/alt/analysis"
sp=pl.read_parquet(f"{O}/sample_pairs.parquet").select("s1","t","label","p")
fd=pl.read_parquet(f"{O}/sample_feat.parquet").select("s1","t","t_blank","name_tset","addr_tset","t_n_s1","t_domain","t_indic","c_other_max","num_conflict")
fb=pl.read_parquet(f"{O}/feat_blank.parquet").select("s1","t","rr_gap","rt_gap","n_claim")
fa=pl.read_parquet(f"{O}/feat_alias.parquet")
fn=pl.read_parquet(f"{O}/feat_noise.parquet").select("s1","t","t_addr_pobox","t_addr_half")
d=(sp.join(fd,on=["s1","t"]).join(fb,on=["s1","t"],how="left").join(fa,on=["s1","t"]).join(fn,on=["s1","t"])
   .with_columns(pl.col("rr_gap","rt_gap").fill_null(0.0), pl.col("n_claim").fill_null(0),
                 lt_core_df=pl.col("t_core_df").log1p(), logit=(pl.col("p").clip(1e-6,1-1e-6)/(1-pl.col("p").clip(1e-6,1-1e-6))).log(),
                 fold=(pl.col("s1").hash(seed=11)%2)))
pe=pl.read_parquet(f"{O}/sample_entity.parquet"); gt=pl.read_parquet(f"{O}/sample_gt.parquet")
BASE=["logit","t_blank","name_tset","addr_tset","t_n_s1","t_domain","t_indic","c_other_max","num_conflict"]
SETS={"base":[], "rr_gap":["rr_gap","rt_gap"], "alias":["lt_core_df","sib_tcore_n","sib_tcore_maxp"],
      "addrjunk":["t_addr_pobox","t_addr_half"], "all":["rr_gap","rt_gap","lt_core_df","sib_tcore_n","sib_tcore_maxp","t_addr_pobox","t_addr_half"]}
zone=(pl.col("p")>0.002)&(pl.col("p")<0.998)
Z=d.filter(zone); print("zone rows",Z.height,"pos",Z["label"].sum())
P={"objective":"binary","learning_rate":0.05,"num_leaves":15,"min_data_in_leaf":50,"verbosity":-1,"num_threads":1,"seed":1}
def macro(dd):
    s=single_owner(prior_shift(dd.select("s1","t","p"),0.6))
    pred=dta_select(s.filter(pl.col("p")>1e-4),0.0).select("s1","t")
    return per_entity(pe.select("s1"),pred,gt)["f"].mean()
print("orig macro", macro(d))
for name,extra in SETS.items():
    cols=BASE+extra; X=Z.select(cols).to_numpy().astype(np.float32); y=Z["label"].to_numpy(); f=Z["fold"].to_numpy()
    oof=np.zeros(len(y))
    for k in (0,1):
        m=lgb.train(P,lgb.Dataset(X[f!=k],y[f!=k]),num_boost_round=300)
        oof[f==k]=m.predict(X[f==k])
    newp=Z.select("s1","t").with_columns(pn=pl.Series(oof))
    dd=d.join(newp,on=["s1","t"],how="left").with_columns(p=pl.coalesce("pn","p"))
    print(f"{name:10s} logloss={log_loss(y,oof):.4f} auc={roc_auc_score(y,oof):.4f} macroF={macro(dd):.5f}")
    if name=="all":
        imp=dict(zip(cols,m.feature_importance("gain"))); print({k:round(v) for k,v in sorted(imp.items(),key=lambda x:-x[1])})
