"""Replace one country's rows of a matching file with pairs from a parquet (s1, t test indices)."""
import sys, polars as pl
base, pairs, out = sys.argv[1], sys.argv[2], sys.argv[3]
W = "/home/user/work_test/test"
s1 = pl.read_parquet(f"{W}/s1.parquet", columns=["s1", "entity_id", "country"]).sort("s1")
tid = pl.read_parquet(f"{W}/tgt.parquet", columns=["t", "entity_id"]).sort("t")["entity_id"]
pr = pl.read_parquet(pairs)
c = s1["country"].gather(pr["s1"]).unique().to_list(); assert len(c) == 1, c
lists = (pr.with_columns(t_id=tid.gather(pr["t"])).group_by("s1").agg(pl.col("t_id").sort().str.join(","))
         .join(s1.select("s1", "entity_id"), on="s1").select(source1_entity_id="entity_id", new="t_id"))
cset = s1.filter(pl.col("country") == c[0]).select(source1_entity_id="entity_id")
m = pl.read_csv(base, separator="\t", quote_char=None, infer_schema=False).fill_null("")
m = (m.join(lists, on="source1_entity_id", how="left", maintain_order="left")
     .with_columns(matched_entity_ids=pl.when(pl.col("source1_entity_id").is_in(cset["source1_entity_id"].implode()))
                   .then(pl.col("new").fill_null("")).otherwise(pl.col("matched_entity_ids"))).drop("new"))
m.write_csv(out, separator="\t", quote_style="never", line_terminator="\n")
print("wrote", out, m.height, "rows; replaced", c[0])
