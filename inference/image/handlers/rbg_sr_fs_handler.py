"""
RBG/SR/FS handler（不加载 diffusers pipeline）
从 inferrer.py::_process_rbg_sr_fs 抽离
"""

from __future__ import annotations

import os
from typing import Any

from common.image_utils import load_image
from common.swap_face_processor import SwapFaceProcessor
from image.runtime.postprocess_pipeline import apply_postprocess
from schemas import InferenceRequestParams


class RbgSrFsHandler:
    def __init__(self, *, inference_config: Any, result_handler: Any, service_id: str, logger: Any):
        self.inference_config = inference_config
        self.result_handler = result_handler
        self.service_id = service_id
        self.logger = logger

    async def run(self, params: InferenceRequestParams) -> None:
        if not params.tpl_list:
            raise ValueError(f"tpl_list is required for job_type {params.job_type}")

        # 约束原始输入图总像素不超过 4096*4096；超过则等比缩小后再处理（RBG/SR/FS 通用）
        # 说明：common.image_utils.load_image 的 max_size 语义是“宽*高不超过 max_size*max_size”
        max_size = 4096
        images = []
        for i, p in enumerate(params.tpl_list):
            img = load_image(p, resize=True, max_size=max_size)
            if img is None:
                self.logger.warning(f"Failed to load tpl_list[{i}]: {p}, skipping")
                continue
            images.append(img)

        if not images:
            raise ValueError("No loadable images in tpl_list")

        total = len(images) if params.job_type in ("RBG", "SR") else len(images)

        # RBG
        if params.job_type == "RBG":
            # RBG 语义：无论 remove_bg 是否显式传入，都必须去背景，并且最后输出 png。
            params.file_type = "png"
            for idx, img in enumerate(images):
                out = apply_postprocess(img, params, force_remove_bg=True)
                await self.result_handler.process_single_result(
                    file_data=out,
                    request_params=params,
                    generate_time=0.0,
                    service_id=self.service_id,
                    file_seed=None,
                    index=idx,
                    total=total,
                )
            return

        # SR
        if params.job_type == "SR":
            for idx, img in enumerate(images):
                out = apply_postprocess(img, params)
                await self.result_handler.process_single_result(
                    file_data=out,
                    request_params=params,
                    generate_time=0.0,
                    service_id=self.service_id,
                    file_seed=None,
                    index=idx,
                    total=total,
                )
            return

        # FS
        if params.job_type == "FS":
            if not params.url:
                raise ValueError("FS requires url as source face image")
            source = load_image(params.url)
            if source is None:
                raise ValueError("FS source face image (url) is not loadable")

            roop_dir = os.path.join(self.inference_config.models_dir, "roop")
            if not os.path.isdir(roop_dir):
                raise FileNotFoundError(
                    f"FS weights dir not found: {roop_dir} (expected '{self.inference_config.models_dir}/roop')"
                )
            proc = SwapFaceProcessor(
                weights_root=roop_dir,
                # 避免与统一后处理链 apply_postprocess(face_enhance=...) 叠加导致“双重增强”。
                # 若确实需要“换脸后立即增强”，可显式设置：
                #   VITOOM_FS_INTERNAL_ENHANCE=1
                use_enhancer=(
                    str(os.getenv("VITOOM_FS_INTERNAL_ENHANCE", "0")).strip().lower() in ("1", "true", "yes", "y", "on")
                )
                and bool(getattr(params, "face_enhance", False)),
            )
            out_images = proc.generate([source] + images)
            for i in range(1, len(out_images)):
                out = apply_postprocess(out_images[i], params)
                await self.result_handler.process_single_result(
                    file_data=out,
                    request_params=params,
                    generate_time=0.0,
                    service_id=self.service_id,
                    file_seed=None,
                    index=i - 1,
                    total=total,
                )
            return

        raise ValueError(f"Unsupported non-pipeline job_type: {params.job_type}")


