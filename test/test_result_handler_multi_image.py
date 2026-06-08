"""
Regression test: multi-image result messages should not be marked completed until the last image.

Why:
- The inferrer sends one `result` message per generated image (files=[...], len=1)
- If each message uses status=completed, frontend may close ws early and miss later images
"""

import asyncio
from types import SimpleNamespace

from PIL import Image

from inference.common.result_handler import ResultHandler
from inference.schemas import InferenceRequestParams


def test_result_handler_multi_image_status(tmp_path):
    async def _run():
        cfg = SimpleNamespace(
            outputs_dir=str(tmp_path),  # LocalBackend destination
            storage_default="local",
            api_base_url="http://127.0.0.1:8888",
        )

        handler = ResultHandler(
            ws_client=None,
            storage_base_path=str(tmp_path),
            inference_config=cfg,
        )

        req = InferenceRequestParams(
            type="image",
            job_type="MK",
            storage="local",
            reference_id="",
            id="msg_1",
            user_id="user_1",
            task_id="task_1",
            prompt="a cat",
            negative_prompt="",
            width=32,
            height=32,
            guidance_scale=7.5,
            seed=1,
            num_inference_steps=1,
            strength=0.5,
            file_type="png",
            url=None,
            generate_num=2,
            model_name="dummy",
            model_id="model_1",
            family="sdxl",
            keep_size="user",
            remove_bg=False,
            fast_mode=True,
            low_vram=False,
            upscale=0,
            face_enhance=False,
            arch="clean",
            duration=0,
            image_file2="",
            edit_act="",
            tpl_list=[],
            model_config=None,
        )

        img = Image.new("RGB", (32, 32), color=(255, 0, 0))

        # first image (index=0 of total=2) -> processing
        resp0 = await handler.process_single_result(
            file_data=img,
            request_params=req,
            generate_time=0.01,
            service_id="service_1",
            file_seed=111,
            index=0,
            total=2,
        )
        assert resp0.status == "processing"
        assert 0 <= resp0.progress < 100
        assert resp0.total == 2

        # second/last image (index=1 of total=2) -> completed
        resp1 = await handler.process_single_result(
            file_data=img,
            request_params=req,
            generate_time=0.01,
            service_id="service_1",
            file_seed=222,
            index=1,
            total=2,
        )
        assert resp1.status == "completed"
        assert resp1.progress == 100
        assert resp1.total == 2

    asyncio.run(_run())


