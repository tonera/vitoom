"""
ControlNet 条件图（controlnet_img / control image）构建器

输入：
- edit_act: 控制类型（canny/openpose/depth/...）
- image_url: 用户上传的“姿态图”（ImageControler.vue -> image_file2）

输出：
- PIL.Image.Image（RGB）作为 ControlNet 条件图

说明：
本仓库目前优先使用 `controlnet_aux`（diffusers 官方推荐的辅助预处理器集合）来生成控制图。
如果运行环境缺少依赖/模型，会抛出带提示的错误，便于部署侧补齐依赖。
"""

from __future__ import annotations

import math
import numpy as np
from pathlib import Path
from typing import Any, Optional

from common.image_utils import load_image


class ControlnetImageBuildError(RuntimeError):
    pass


def _resize_to_target_area(img: Any, *, target_area: int = 1024 * 1024, multiple: int = 8) -> Any:
    """
    对齐 xinsir 官方示例：
    ratio = sqrt(target_area / (w*h))
    new_w/new_h = int(w*ratio), int(h*ratio)
    额外：收敛到 multiple 倍数，避免 SDXL latent 尺寸报错。
    """
    try:
        w, h = getattr(img, "size", (0, 0))
        w, h = int(w), int(h)
    except Exception:
        w, h = 0, 0
    if w <= 0 or h <= 0:
        return img

    ratio = math.sqrt(float(target_area) / float(w * h))
    new_w = max(multiple, int(w * ratio))
    new_h = max(multiple, int(h * ratio))
    new_w = max(multiple, (new_w // multiple) * multiple)
    new_h = max(multiple, (new_h // multiple) * multiple)

    try:
        from PIL import Image  # type: ignore

        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    except Exception:
        # 没 PIL 就原样返回（上层会报更明确的错误）
        return img


def _resize_hw_to_target_area(w: int, h: int, *, target_area: int = 1024 * 1024, multiple: int = 8) -> tuple[int, int]:
    """
    仅计算 resize 后的宽高（对齐官方示例的 area-based resize），并收敛到 multiple 倍数。
    """
    w = int(w or 0)
    h = int(h or 0)
    if w <= 0 or h <= 0:
        return w, h
    ratio = math.sqrt(float(target_area) / float(w * h))
    new_w = max(multiple, int(w * ratio))
    new_h = max(multiple, int(h * ratio))
    new_w = max(multiple, (new_w // multiple) * multiple)
    new_h = max(multiple, (new_h // multiple) * multiple)
    return new_w, new_h


def union_control_type_index(edit_act: str) -> int:
    """
    xinsir ControlNetUnion 的 control type index（6 类）：
    0 -- openpose
    1 -- depth
    2 -- hed/pidi/scribble/ted
    3 -- canny/lineart/anime_lineart/mlsd
    4 -- normal
    5 -- segment
    """
    act = (edit_act or "").strip().lower()
    # 兼容前端/协议：pose == openpose
    if act in ("openpose", "pose"):
        return 0
    if act == "depth":
        return 1
    if act in ("hed", "pidi", "scribble", "ted"):
        return 2
    if act in ("canny", "lineart", "anime_lineart", "mlsd"):
        return 3
    if act == "normal":
        return 4
    if act == "segment":
        return 5
    raise ControlnetImageBuildError(f"Unsupported edit_act for ControlNetUnion: {edit_act}")


def build_controlnet_image(
    edit_act: str,
    image_url: str,
    *,
    weights_dir: Optional[str] = None,
    logger: Any = None,
) -> Any:
    """
    根据 edit_act 生成 ControlNet 条件图。
    当前优先覆盖 UI 已接入的三类：canny / openpose / depth。
    其他类型会尝试使用 controlnet_aux 对应 detector；若环境缺依赖则抛错。
    """
    act = (edit_act or "").strip().lower()
    # 兼容前端/协议：pose == openpose
    if act == "pose":
        act = "openpose"
    if not image_url:
        raise ControlnetImageBuildError("POSE requires image_file2 as control source image")

    img = load_image(image_url, resize=False)
    if img is None:
        raise ControlnetImageBuildError(f"Control image (image_file2) is not loadable: {image_url}")

    # 统一转 RGB（detector 通常要求）
    try:
        if getattr(img, "mode", "") != "RGB":
            img = img.convert("RGB")
    except Exception:
        pass
    img_src = img

    def _log(msg: str):
        if logger is not None:
            try:
                logger.info(msg)
            except Exception:
                pass

    # 本项目部署约定：controlnet_aux 预处理器权重在 {weights_dir}/Annotators
    # 兼容官方示例的 repo id：
    # - OpenPose: lllyasviel/ControlNet
    # - 其他 detector: lllyasviel/Annotators
    annotators_repo: str = "lllyasviel/Annotators"
    openpose_repo: str = "lllyasviel/ControlNet"
    if weights_dir:
        p = Path(str(weights_dir)).expanduser().resolve() / "Annotators"
        if not p.exists():
            raise ControlnetImageBuildError(f"Annotators directory not found: {p}")
        # 约定本地目录已包含对应权重（无论源自 Annotators 还是 ControlNet repo）
        annotators_repo = str(p)
        openpose_repo = str(p)

    # detector 构建（按需懒加载）
    if act == "canny":
        # canny 走本地实现，避免强依赖 controlnet_aux（更稳定）
        _log("POSE: building canny control image via local canny (cv2 if available), resize-to-1MP then canny (matches official)")
        arr = np.array(img_src)
        try:
            import cv2  # type: ignore

            # 对齐官方示例：先 resize 到 1MP，再做 Canny
            h, w = arr.shape[:2]
            new_w, new_h = _resize_hw_to_target_area(w, h, target_area=1024 * 1024, multiple=8)
            if new_w > 0 and new_h > 0 and (new_w != w or new_h != h):
                arr = cv2.resize(arr, (new_w, new_h))

            edges = cv2.Canny(arr, 100, 200)  # HxW
            # 对齐官方 HWC3：确保输出为 3 通道 uint8
            if edges.ndim == 2:
                edges = edges[:, :, None]
            if edges.shape[2] == 1:
                edges = np.concatenate([edges, edges, edges], axis=2)
            from PIL import Image  # type: ignore

            return Image.fromarray(edges)
        except Exception:
            # 无 cv2 兜底：简单边缘检测（效果不如 canny，但能跑通）
            try:
                from PIL import Image, ImageFilter  # type: ignore

                # 尽量对齐官方：先 resize 再做边缘（PIL 兜底）
                img2 = _resize_to_target_area(img_src, target_area=1024 * 1024, multiple=8)
                edge = img2.filter(ImageFilter.FIND_EDGES).convert("L")
                edge_arr = np.array(edge)
                edge_arr = (edge_arr > 32).astype(np.uint8) * 255
                edge_arr = np.stack([edge_arr, edge_arr, edge_arr], axis=2)
                return Image.fromarray(edge_arr)
            except Exception as e:
                raise ControlnetImageBuildError("Failed to build canny control image (cv2/PIL not available)") from e

    # 其他类型依赖 controlnet_aux（部署侧需安装并准备 annotator 权重）
    try:
        import controlnet_aux  # type: ignore
    except Exception as e:
        raise ControlnetImageBuildError(
            "Missing dependency: controlnet_aux. Please install it in inference environment "
            "to enable POSE control image preprocessing for openpose/depth/others."
        ) from e

    if act == "openpose":
        # 对齐官方示例：
        # 1) 在原图上做 pose 检测（不要先缩放，避免骨架细节丢失）
        # 2) hand_and_face=False
        # 3) 输出姿态图后再缩放到 1MP 面积
        try:
            detector = getattr(controlnet_aux, "OpenposeDetector").from_pretrained(openpose_repo)
        except Exception as e:
            raise ControlnetImageBuildError(
                f"controlnet_aux.OpenposeDetector not available or annotator weights missing (expected '{openpose_repo}')"
            ) from e
        _log("POSE: building openpose control image via controlnet_aux.OpenposeDetector")
        try:
            # 官方示例用 cv2(BGR) 输入；这里对齐一下，避免色彩空间差异影响 detector。
            bgr = np.array(img_src)[:, :, ::-1]
            out = detector(bgr, hand_and_face=False, output_type="pil")
        except TypeError:
            # 兼容旧版本 controlnet_aux（可能不支持 output_type 参数）
            out = detector(img_src)
        return _resize_to_target_area(out, target_area=1024 * 1024, multiple=8)

    if act == "depth":
        # 对齐官方示例：优先 ZoeDetector，其次 MidasDetector；先 detector 再 resize 到 1MP
        detector = None
        det_name = None
        try:
            detector = getattr(controlnet_aux, "ZoeDetector").from_pretrained(annotators_repo)
            det_name = "ZoeDetector"
        except Exception:
            detector = None
        if detector is None:
            try:
                detector = getattr(controlnet_aux, "MidasDetector").from_pretrained(annotators_repo)
                det_name = "MidasDetector"
            except Exception as e:
                raise ControlnetImageBuildError(
                    f"controlnet_aux depth detector not available or weights missing (expected '{annotators_repo}')"
                ) from e

        _log(f"POSE: building depth control image via controlnet_aux.{det_name}")
        try:
            bgr = np.array(img_src)[:, :, ::-1]
            out = detector(bgr, output_type="pil")
        except TypeError:
            # 兼容旧版本 controlnet_aux
            out = detector(img_src)
        return _resize_to_target_area(out, target_area=1024 * 1024, multiple=8)

    # 其余类型：先尽力映射；若 detector 不存在/不可用，则报错（便于后续按需补齐）
    detector_name: Optional[str] = None
    if act == "hed":
        detector_name = "HEDdetector"
    elif act == "pidi":
        detector_name = "PidiNetDetector"
    elif act == "scribble":
        detector_name = "HEDdetector"  # 先用 HED 近似；更精细的 scribble 可后续再做阈值化
    elif act == "ted":
        detector_name = "TEEDdetector"
    elif act == "lineart":
        detector_name = "LineartDetector"
    elif act == "anime_lineart":
        detector_name = "LineartAnimeDetector"
    elif act == "mlsd":
        detector_name = "MLSDdetector"
    elif act == "normal":
        detector_name = "NormalBaeDetector"
    elif act == "segment":
        detector_name = "UniformerDetector"

    if detector_name:
        try:
            Det = getattr(controlnet_aux, detector_name)
            # 多数 detector 支持 from_pretrained("lllyasviel/Annotators")
            detector = Det.from_pretrained(annotators_repo) if hasattr(Det, "from_pretrained") else Det()
        except Exception as e:
            raise ControlnetImageBuildError(f"controlnet_aux.{detector_name} not available or weights missing") from e
        _log(f"POSE: building {act} control image via controlnet_aux.{detector_name}")
        return detector(img)

    raise ControlnetImageBuildError(f"Unsupported edit_act: {edit_act}")


