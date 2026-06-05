import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.agent.tools.builtin.analyze_media import _do_analyze_media


def test_do_analyze_media_passes_preferred_model_name():
    with patch(
        "backend.services.agent.tools.builtin.analyze_media.run_multimodal_completion",
        return_value={"content": "图中是一只猫"},
    ) as mocked:
        result = _do_analyze_media(
            effective_user_id="user-1",
            preferred_model_name="qwen3.6:35b",
            url="https://example.com/cat.jpg",
            question="图里是什么？",
        )

    assert result == "图中是一只猫"
    mocked.assert_called_once_with(
        user_id="user-1",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "图里是什么？"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.jpg"},
                    },
                ],
            }
        ],
        model="qwen3.6:35b",
    )
