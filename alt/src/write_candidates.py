"""Write candidate_pairs.tsv (the exact pairs the matcher scored) in S1 chunks to bound memory."""
import sys
import polars as pl
work, out = sys.argv[1], sys.argv[2]
W = f"{work}/test"
s1 = pl.read_parquet(f"{W}/s1.parquet", columns=["s1", "entity_id"]).sort("s1")
tid = pl.read_parquet(f"{W}/tgt.parquet", columns=["t", "entity_id"]).sort("t")["entity_id"]
sc = pl.read_parquet(f"{W}/scores_c.parquet", columns=["s1", "t"])
step = 100_000
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("source1_entity_id\tcandidate_entity_ids\n")
    for lo in range(0, s1.height, step):
        part = sc.filter(pl.col("s1").is_between(lo, lo + step - 1))
        lists = (part.with_columns(t_id=tid.gather(part["t"])).group_by("s1")
                 .agg(pl.col("t_id").unique().sort().str.join(",")))
        rows = s1.slice(lo, step).join(lists, on="s1", how="left", maintain_order="left").with_columns(pl.col("t_id").fill_null(""))
        fh.write("".join(f"{a}\t{b}\n" for a, b in zip(rows["entity_id"], rows["t_id"])))
print("written", out)
