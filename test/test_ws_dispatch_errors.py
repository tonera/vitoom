"""WS 派发失败 message_code 推断单测。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.i18n.ws_messages import (  # noqa: E402
    enrich_task_ws_message,
    infer_message_code_and_params_from_error,
)


def test_infer_dispatch_failure_with_load_name():
    error = (
        "No running inference service available for task_type=audio capability=asr "
        "(requested load_name=Qwen3-ASR-1.7B is not served by any connected+running audio service)"
    )
    code, params = infer_message_code_and_params_from_error(error)
    assert code == "inference.modelNotAvailable"
    assert params == {"model": "Qwen3-ASR-1.7B"}


def test_enrich_failed_task_prefers_model_not_available():
    enriched = enrich_task_ws_message(
        {
            "type": "task_status",
            "status": "failed",
            "error": (
                "No running inference service available for task_type=audio capability=asr "
                "(requested load_name=Qwen3-ASR-1.7B is not served by any connected+running audio service)"
            ),
        }
    )
    assert enriched["message_code"] == "inference.modelNotAvailable"
    assert enriched["message_params"] == {"model": "Qwen3-ASR-1.7B"}


def test_enrich_failed_task_keeps_explicit_message_code():
    enriched = enrich_task_ws_message(
        {
            "type": "task_status",
            "status": "failed",
            "message_code": "inference.modelNotAvailable",
            "message_params": {"model": "Demo"},
        }
    )
    assert enriched["message_code"] == "inference.modelNotAvailable"
    assert enriched["message_params"] == {"model": "Demo"}
