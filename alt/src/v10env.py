"""Shared paths: v10 source on sys.path, v10 artifacts in work_v10, raw data."""
import os, sys
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V10_SRC = os.path.join(ROOT, "v10", "code_v10", "code", "business_entity_resolution", "src")
DATA = os.path.join(ROOT, "student_resource", "dataset")
WORK = os.environ.get("ALT_WORK", "/home/user/work_v10")
sys.path.insert(0, V10_SRC)


def rd(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def s1_train():
    """Train S1 in file order: s1 index, entity_id, country (v10's dense index = row order)."""
    return rd(f"{DATA}/train/train_source1.tsv").with_row_index("s1").select("s1", "entity_id", "country")


def tgt_train():
    """Train targets: S2 rows then S3 rows (v10's dense t index)."""
    return pl.concat([rd(f"{DATA}/train/train_source{i}.tsv") for i in (2, 3)]).with_row_index("t")
