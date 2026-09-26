import polars as pl
D="student_resource/dataset"
def rd(p): return pl.read_csv(p,separator="\t",quote_char=None,infer_schema=False)
def nums(df): return df.with_columns(n=pl.col("business_address").fill_null("").str.to_lowercase().str.replace_all(r"\b(bis|ter)\b","").str.extract_all(r"\d+").list.eval(pl.element().cast(pl.Int64,strict=False).cast(pl.String)).list.unique())
def ev(sp,pairs_path,col):
    s1=nums(rd(f"{D}/{sp}/{sp}_source1.tsv")).select("entity_id","country",sn="n")
    tg=nums(pl.concat([rd(f"{D}/{sp}/{sp}_source{i}.tsv") for i in (2,3)])).select("entity_id",tn="n")
    p=rd(pairs_path).rename({col:"m"}).with_columns(t=pl.col("m").str.split(",")).explode("t").filter(pl.col("t")!="")
    j=p.join(s1,left_on="source1_entity_id",right_on="entity_id").join(tg,left_on="t",right_on="entity_id")
    j=j.with_columns(both=(pl.col("sn").list.len()>0)&(pl.col("tn").list.len()>0),share=pl.col("sn").list.set_intersection("tn").list.len()>0)
    print(sp, pairs_path)
    print(j.group_by("country").agg(pairs=pl.len(),both_have_num=pl.col("both").mean(),share_num_given_both=pl.col("share").filter(pl.col("both")).mean()).sort("country"))
ev("train",f"{D}/train/train_ground_truth.tsv","matched_entity_ids")
ev("test","v10/matching_results.tsv","matched_entity_ids")
