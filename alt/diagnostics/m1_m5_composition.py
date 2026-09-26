import polars as pl
D="student_resource/dataset"
def rd(p): return pl.read_csv(p,separator="\t",quote_char=None,infer_schema=False,missing_utf8_is_empty_string=True)
for sp in ["train","test"]:
    s=[rd(f"{D}/{sp}/{sp}_source{i}.tsv") for i in (1,2,3)]
    c1=s[0].group_by("country").len().rename({"len":"s1"})
    c2=pl.concat(s[1:]).group_by("country").len().rename({"len":"tgt"})
    t=c1.join(c2,on="country",how="full",coalesce=True).with_columns(per_s1=pl.col("tgt")/pl.col("s1"),s1_share=pl.col("s1")/pl.col("s1").sum())
    print(sp);print(t)
    if sp=="train":
        gt=rd(f"{D}/train/train_ground_truth.tsv").with_columns(n=pl.when(pl.col("matched_entity_ids")=="").then(0).otherwise(pl.col("matched_entity_ids").str.count_matches(",")+1))
        g=gt.join(s[0].select("entity_id","country"),left_on="source1_entity_id",right_on="entity_id")
        print(g.group_by("country").agg(owned_per_s1=pl.col("n").mean(),singleton=(pl.col("n")==0).mean()))
    else:
        v=rd("v10/matching_results.tsv").with_columns(n=pl.when(pl.col("matched_entity_ids")=="").then(0).otherwise(pl.col("matched_entity_ids").str.count_matches(",")+1))
        g=v.join(s[0].select("entity_id","country"),left_on="source1_entity_id",right_on="entity_id")
        print("v10 test predictions");print(g.group_by("country").agg(pred_per_s1=pl.col("n").mean(),empty=(pl.col("n")==0).mean(),n1=(pl.col("n")==1).mean()))
