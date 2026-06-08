from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from test.text_ws_experiment import summarize_turn_stats


def test_summarize_turn_stats_uses_local_timing_without_token_fallback():
    stats = summarize_turn_stats(
        started_at=10.0,
        first_delta_at=10.5,
        finished_at=12.5,
    )

    assert stats["input_tokens"] is None
    assert stats["output_tokens"] is None
    assert stats["ttft_seconds"] == 0.5
    assert stats["total_seconds"] == 2.5
    assert stats["decode_seconds"] == 2.0
    assert stats["output_tps_total"] is None
    assert stats["output_tps_decode"] is None


def test_summarize_turn_stats_prefers_server_side_stats():
    stats = summarize_turn_stats(
        started_at=10.0,
        first_delta_at=10.5,
        finished_at=12.5,
        server_stats={
            "prompt_tokens": 7,
            "output_tokens": 15,
            "ttft_seconds": 0.4,
            "total_seconds": 2.0,
            "decode_seconds": 1.6,
            "tok_s_total": 7.5,
            "tok_s_decode": 9.375,
        },
    )

    assert stats["input_tokens"] == 7
    assert stats["output_tokens"] == 15
    assert stats["ttft_seconds"] == 0.4
    assert stats["total_seconds"] == 2.0
    assert stats["decode_seconds"] == 1.6
    assert stats["output_tps_total"] == 7.5
    assert stats["output_tps_decode"] == 9.375
