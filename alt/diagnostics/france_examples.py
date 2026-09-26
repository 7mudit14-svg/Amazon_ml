import polars as pl
D="student_resource/dataset"
def rd(p): return pl.read_csv(p,separator="\t",quote_char=None,infer_schema=False)
s1=rd(f"{D}/test/test_source1.tsv").filter(pl.col("country")=="France")
tg=pl.concat([rd(f"{D}/test/test_source{i}.tsv") for i in (2,3)]).filter(pl.col("country")=="France")
v=rd("v10/matching_results.tsv").join(s1.select("entity_id"),left_on="source1_entity_id",right_on="entity_id",how="semi")
pairs=v.with_columns(t=pl.col("matched_entity_ids").str.split(",")).explode("t").filter(pl.col("t")!="")
# first word of name as crude family key
fw=lambda df: df.with_columns(w=pl.col("business_name").str.to_lowercase().str.extract(r"([a-zà-ÿ]{4,})"))
S=fw(s1).sample(6,seed=3)
T=fw(tg)
pd=set(pairs["t"].to_list())
for r in S.iter_rows(named=True):
    print("\n=== S1",r["entity_id"],"|",r["business_name"],"|",r["business_address"])
    m=T.filter(pl.col("w")==r["w"]).head(40)
    pr=set(pairs.filter(pl.col("source1_entity_id")==r["entity_id"])["t"].to_list())
    for x in m.iter_rows(named=True):
        print("  ","MATCH" if x["entity_id"] in pr else ("other" if x["entity_id"] in pd else "  -  "),x["entity_id"][:3],"|",x["business_name"],"|",x["business_address"])
