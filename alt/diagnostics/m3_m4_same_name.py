import polars as pl
D="student_resource/dataset"
def rd(p): return pl.read_csv(p,separator="\t",quote_char=None,infer_schema=False)
LEG=r"\b(pvt|private|ltd|limited|llc|llp|inc|incorporated|corp|corporation|co|company|plc|lp|pc|sarl|sas|sasu|sa|eurl|sci|snc|ei|public|group|the|and|of)\b"
def key(df):
    return df.with_columns(k=pl.col("business_name").str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]"," ").str.replace_all(LEG," ").str.split(" ").list.eval(pl.element().filter(pl.element()!="")).list.sort().list.join(" "))
for sp in ["train","test"]:
    s1=key(rd(f"{D}/{sp}/{sp}_source1.tsv")).filter(pl.col("k")!="")
    tg=key(pl.concat([rd(f"{D}/{sp}/{sp}_source{i}.tsv") for i in (2,3)])).filter(pl.col("k")!="")
    tk=tg.group_by("country","k").agg(nt=pl.len())
    sk=s1.group_by("country","k").agg(ns=pl.len())
    j=s1.join(tk,on=["country","k"],how="left").join(sk,on=["country","k"]).with_columns(pl.col("nt").fill_null(0))
    print(sp,"per S1 with unique name: same-name targets")
    print(j.filter(pl.col("ns")==1).group_by("country").agg(mean_same_name_tgt=pl.col("nt").mean(),p0=(pl.col("nt")==0).mean(),q50=pl.col("nt").median(),q90=pl.col("nt").quantile(0.9)))
    # targets whose name matches no S1 at all
    orphan=tg.join(sk,on=["country","k"],how="anti").group_by("country").len()
    print("targets whose exact name key matches no S1:",orphan.join(tg.group_by("country").agg(N=pl.len()),on="country").with_columns(share=pl.col("len")/pl.col("N")))
    if sp=="train":
        gt=rd(f"{D}/train/train_ground_truth.tsv").with_columns(t=pl.col("matched_entity_ids").str.split(",")).explode("t").filter(pl.col("t")!="")
        own=tg.join(gt.select("t"),left_on="entity_id",right_on="t",how="semi")
        print("owned targets with exact key == owner's key share:",
              gt.join(s1.select("entity_id",sk_="k"),left_on="source1_entity_id",right_on="entity_id").join(tg.select("entity_id",tk_="k"),left_on="t",right_on="entity_id").select((pl.col("sk_")==pl.col("tk_")).mean()))
        un=tg.join(gt.select("t"),left_on="entity_id",right_on="t",how="anti")
        print("UNOWNED targets: share with same-name S1:",un.join(sk,on=["country","k"],how="semi").height/un.height, "n unowned",un.height)
