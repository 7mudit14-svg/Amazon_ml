import polars as pl
import pytest

from data_io import MATCH_HEADER, write_id_lists
from make_submission import _read_back
from model_a import decide


def _scores(rows):
    return pl.DataFrame(rows, schema={"s1": pl.UInt32, "t": pl.UInt32, "p": pl.Float32}, orient="row")


def test_threshold_and_cap():
    scores = _scores([(0, 1, 0.9), (0, 2, 0.4), (1, 3, 0.2)])
    out = decide(scores, tau=0.5, rho=0.0, excl=False)
    assert out.sort("t").rows() == [(0, 1)]


def test_target_exclusivity_keeps_best_owner():
    scores = _scores([(0, 7, 0.9), (1, 7, 0.6), (1, 8, 0.7)])
    out = decide(scores, tau=0.5, rho=0.0, excl=True)
    assert sorted(out.rows()) == [(0, 7), (1, 8)]


def test_relative_cut():
    scores = _scores([(0, 1, 0.95), (0, 2, 0.55)])
    assert sorted(decide(scores, tau=0.5, rho=0.8, excl=False).rows()) == [(0, 1)]
    assert sorted(decide(scores, tau=0.5, rho=0.5, excl=False).rows()) == [(0, 1), (0, 2)]


def test_at_most_eleven_matches():
    scores = _scores([(0, t, 0.9) for t in range(20)])
    assert decide(scores, tau=0.5, rho=0.0, excl=False).height == 11


def test_write_and_read_back(tmp_path):
    order = pl.Series(["S1-1", "S1-2", "S1-3"])
    pairs = pl.DataFrame({"s1_id": ["S1-1", "S1-1", "S1-3"], "t_id": ["S3-9", "S2-5", "S2-5"]})
    path = tmp_path / "m.tsv"
    write_id_lists(path, MATCH_HEADER, order, pairs)
    assert path.read_text(encoding="utf-8") == (
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-5,S3-9\nS1-2\t\nS1-3\tS2-5\n"
    )
    got = _read_back(path, MATCH_HEADER, order.to_list(), {"S2-5", "S3-9"})
    assert got == {"S1-1": {"S2-5", "S3-9"}, "S1-2": set(), "S1-3": {"S2-5"}}


def test_read_back_rejects_unknown_ids(tmp_path):
    path = tmp_path / "m.tsv"
    path.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-404\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        _read_back(path, MATCH_HEADER, ["S1-1"], {"S2-5"})
