"""Tests für Reciprocal Rank Fusion (deterministisch, score-invariant)."""

from __future__ import annotations

from rag_module.models import ScoredChunk
from rag_module.retrieval import reciprocal_rank_fusion


def _chunk(point_id: str, score: float) -> ScoredChunk:
    return ScoredChunk(point_id=point_id, score=score, payload={"content": point_id})


def test_rrf_prefers_documents_ranked_high_in_multiple_lists() -> None:
    dense = [_chunk("both", 0.9), _chunk("dense-only", 0.8)]
    sparse = [_chunk("both", 12.0), _chunk("sparse-only", 4.0)]

    fused = reciprocal_rank_fusion([dense, sparse], k=60)

    assert fused[0].point_id == "both"
    assert fused[0].hit_count == 2
    assert {candidate.point_id for candidate in fused} == {
        "both",
        "dense-only",
        "sparse-only",
    }


def test_rrf_is_invariant_to_score_scale() -> None:
    small_scores = [_chunk("a", 0.01), _chunk("b", 0.001)]
    huge_scores = [_chunk("b", 1000.0), _chunk("c", 900.0)]

    fused = reciprocal_rank_fusion([small_scores, huge_scores], k=60)

    # "b" gewinnt über Ränge (Rang 2 + Rang 1), nicht über Roh-Scores.
    assert fused[0].point_id == "b"


def test_rrf_tie_break_is_deterministic() -> None:
    first = [_chunk("x", 1.0)]
    second = [_chunk("y", 1.0)]

    fused_a = reciprocal_rank_fusion([first, second], k=60)
    fused_b = reciprocal_rank_fusion([first, second], k=60)

    assert [c.point_id for c in fused_a] == [c.point_id for c in fused_b]
    # Gleicher RRF-Score, gleicher bester Rang -> Punkt-ID entscheidet stabil.
    assert [c.point_id for c in fused_a] == ["x", "y"]


def test_rrf_empty_input() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
