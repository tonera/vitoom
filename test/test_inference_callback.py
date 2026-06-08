import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 允许直接运行该脚本时，也能正确导入 inference 下的模块（schemas/image/common）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "inference"))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="手动触发 ImageInferrer.inference_callback 的测试脚本（支持命令行传参）"
    )

    # InferenceRequestParams 的必填/关键字段
    p.add_argument("--type", default="image", help="任务大类（默认: image）")
    p.add_argument("--job-type", default="MK", help="任务执行分类（默认: MK）")
    p.add_argument("--storage", default="local", choices=["local", "oss", "s3", "server"], help="存储方式")
    p.add_argument("--id", default="task1", help="消息标识（默认: task1）")
    p.add_argument("--user-id", default="u1", help="用户ID（默认: u1）")
    p.add_argument("--task-id", default="task1", help="任务ID（默认: task1）")

    # 你标注的参数（原脚本 7-22 行）
    p.add_argument(
        "--model-name",
        default="xl-base",
        help="模型名称（不含路径；将自动拼接到 inference_config.models_dir 下）",
    )
    p.add_argument("--prompt", default="A beautiful girl", help="提示词")
    p.add_argument("--num-inference-steps", type=int, default=20, help="推理步数")
    p.add_argument("--guidance-scale", type=float, default=7.5, help="引导比例")
    p.add_argument("--negative-prompt", default="", help="负面提示词")
    p.add_argument("--width", type=int, default=512, help="图片宽度")
    p.add_argument("--height", type=int, default=512, help="图片高度")
    p.add_argument("--file-type", default="jpeg", choices=["jpeg", "png", "webp"], help="输出格式")

    # 原脚本 num_images（真实字段是 generate_num，这里做兼容映射）
    p.add_argument("--num-images", type=int, default=1, help="生成图片数量（映射到 generate_num）")

    p.add_argument("--seed", type=int, default=42, help="随机种子（<=0 表示随机）")
    p.add_argument("--fast-mode", action=argparse.BooleanOptionalAction, default=False, help="快速模式")
    p.add_argument(
        "--pretouch-cpu-tensors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="诊断用：在 pipe.to(cuda) 前调用 pretouch_pipeline_cpu_tensors，并打印耗时",
    )
    p.add_argument(
        "--low-vram",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="低显存模式（diffusers cpu_offload / 相关优化），默认不启用",
    )

    # 新增：后处理 / 批处理输入
    p.add_argument(
        "--tpl-list",
        default="",
        help="批处理图片列表：用 '||' 分隔，例如: '/abs/a.jpg||/abs/b.jpg'（也兼容JSON: '[\"/abs/a.jpg\",\"/abs/b.jpg\"]'）",
    )
    p.add_argument("--remove-bg", action=argparse.BooleanOptionalAction, default=False, help="生成后去背景（最后执行，强制png）")
    p.add_argument("--upscale", type=int, default=0, choices=[0, 1, 2, 4], help="超分倍数（仅2/4生效）")
    p.add_argument(
        "--face-enhance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="人脸增强（默认 GFPGAN；可通过 VITOOM_FACE_ENHANCER 切换 CodeFormer）",
    )
    p.add_argument("--arch", default="clean", help="人脸增强架构: clean/original/RestoreFormer")

    # model_cfg 仅包含两个键：transformer/unet，直接从命令行传参即可
    p.add_argument("--transformer", default="", help="model_cfg.transformer")
    p.add_argument("--nunchaku-transformer", default="", help="model_cfg.nunchaku_transformer")
    p.add_argument("--unet", default="", help="model_cfg.unet")
    p.add_argument(
        "--prediction-type",
        default="",
        choices=["", "epsilon", "v_prediction"],
        help="（仅SDXL）model_cfg.prediction_type：指定后会跳过 v_prediction 自动探测（quick-latent-test）",
    )

    # 可选：图生图 URL
    p.add_argument("--url", default=None, help="图生图 URL（可选）")

    # 可选：service_id
    p.add_argument("--service-id", default=os.getenv("VITOOM_SERVICE_ID", "service_123"), help="service_id（可选）")
    return p

def test_inference_callback():
    """
    pytest 入口：保持默认参数可跑（需要本地环境/模型齐全）。

    从命令行传参请运行：
      python test/test_inference_callback.py --help
    """
    args = _build_arg_parser().parse_args([])
    asyncio.run(_run_from_args(args))


async def _run_from_args(args: argparse.Namespace) -> None:
    # 延迟导入：保证 `python test/test_inference_callback.py --help` 不依赖 torch 等重型依赖
    from image.inferrer import ImageInferrer  # type: ignore[import-not-found]
    from schemas import InferenceRequestParams  # type: ignore[import-not-found]

    # 诊断：该脚本常见卡点是 pipeline 加载/CPU pretouch/迁移 CUDA。
    # 这里按 test/single_file_zimage.py 的输出风格，在 pipe.to(cuda) 前后增加时间打印。
    try:
        import image.runtime.pipeline_lifecycle as _pl  # type: ignore[import-not-found]
        import image.runtime.pipeline_service as _ps  # type: ignore[import-not-found]

        _orig_move_to_device = getattr(_pl.PipelineLifecycle, "move_to_device", None)
        if callable(_orig_move_to_device):

            def _timed_move_to_device(self, pipe, plan, params):
                if getattr(args, "pretouch_cpu_tensors", False):
                    try:
                        from common.torch_transfer_utils import pretouch_pipeline_cpu_tensors

                        print(
                            f"开始 pretouch_pipeline_cpu_tensors,当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        t0 = time.time()
                        pretouch_pipeline_cpu_tensors(pipe, on_component=lambda n: print(f"{n} pretouch完成"))
                        dt = time.time() - t0
                        print(
                            f"pretouch_pipeline_cpu_tensors完成,当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}，耗时: {dt:.2f}s"
                        )
                    except Exception as e:
                        print(f"pretouch_pipeline_cpu_tensors失败（忽略继续）,当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}，错误: {e}")

                # pipe.to(target) 的耗时通常是最大卡点
                print(f"开始 pipeline.to(device),当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                t1 = time.time()
                out = _orig_move_to_device(self, pipe, plan, params)
                dt2 = time.time() - t1
                print(
                    f"pipeline.to(device)完成,当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}，耗时: {dt2:.2f}s"
                )
                return out

            _pl.PipelineLifecycle.move_to_device = _timed_move_to_device  # type: ignore[assignment]

        # 诊断：计时 pipeline 创建（from_pretrained/from_single_file）阶段
        _orig_acquire = getattr(_ps.PipelineService, "acquire", None)
        if callable(_orig_acquire):

            async def _timed_acquire(self, params):
                print(f"开始 PipelineService.acquire,当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                t0 = time.time()
                out = await _orig_acquire(self, params)
                dt = time.time() - t0
                print(
                    f"PipelineService.acquire完成,当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}，耗时: {dt:.2f}s"
                )
                return out

            _ps.PipelineService.acquire = _timed_acquire  # type: ignore[assignment]
    except Exception:
        # 诊断代码：不影响主流程
        pass

    inferrer = ImageInferrer(args.service_id)
    # 需要初始化：建立 ws_client/task_processor/result_handler 等
    await inferrer.initialize()
    model_cfg = {"transformer": args.transformer, "unet": args.unet, "nunchaku_transformer": args.nunchaku_transformer}
    if getattr(args, "prediction_type", ""):
        model_cfg["prediction_type"] = args.prediction_type

    # tpl_list：支持 "a||b||c" 或 JSON 字符串
    tpl_list = []
    if args.tpl_list:
        s = str(args.tpl_list).strip()
        if s.startswith("["):
            try:
                tpl_list = json.loads(s)
            except Exception:
                tpl_list = []
        else:
            tpl_list = [x.strip() for x in s.split("||") if x.strip()]

    params = InferenceRequestParams(
        # required
        type=args.type,
        job_type=args.job_type,
        storage=args.storage,
        id=args.id,
        user_id=args.user_id,
        task_id=args.task_id,
        # inference
        model_name=args.model_name,
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        seed=args.seed,
        generate_num=args.num_images,
        fast_mode=args.fast_mode,
        low_vram=args.low_vram,
        url=args.url,
        tpl_list=tpl_list,
        file_type=args.file_type,
        remove_bg=args.remove_bg,
        upscale=args.upscale,
        face_enhance=args.face_enhance,
        arch=args.arch,
        model_cfg=model_cfg,
    )
    await inferrer.inference_callback(params)



if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    asyncio.run(_run_from_args(args))