"""人脸合成 paste-back（vendored from LiveTalking ``avatars/musetalk/utils/blending.py``）。

唯一 API ``get_image_blending`` —— 把推理出来的人脸 patch（``face``）按 mask
回贴到原始全身帧（``image``）上。

实现说明：
    生成端 ``get_image_prepare_material`` 用 PIL ``body.crop(crop_box)``
    计算 mask，PIL crop 在 crop_box 越出原图边界时会**自动用 0 扩展画布**，
    所以 mask 大小始终是 ``(y_e - y_s, x_e - x_s)``。

    早先这里曾用 numpy 切片 ``body[y_s:y_e, x_s:x_e]`` 做"快速路径"，但
    人脸靠近画面边缘时 ``get_crop_box`` 会扩出**负数**坐标，numpy 负索引
    会被解释为"从尾部回数"，导致切片反向退化到 0 宽 —— ``paste_back_frame``
    刷屏 ``could not broadcast input array from shape (...) into shape
    (...,0,3)`` 的根因就是这个。

    现在改回与生成端一致的 PIL 实现，PIL paste(image, box, mask) 能正确
    处理负坐标与图像边界，对 25fps 推理性能影响可忽略。
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def get_image_blending(image, face, face_box, mask_array, crop_box):
    # PIL 内部走 RGB；cv2 ndarray 是 BGR
    body = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

    x, y, x1, y1 = face_box
    x_s, y_s, _, _ = crop_box
    face_large = body.crop(crop_box)  # 越界自动用 0 扩展

    mask_image = Image.fromarray(mask_array).convert("L")  # mask 来自 cv2.imread (BGR) -> L
    face_large.paste(face_pil, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    body.paste(face_large, crop_box[:2], mask_image)
    # 用 cvtColor 而不是 [:, :, ::-1]：后者产生 negative-stride view，不 contiguous，
    # 会让下游 cv2.putText 报 "Layout of the output array img is incompatible with cv::Mat"。
    return cv2.cvtColor(np.array(body), cv2.COLOR_RGB2BGR)


__all__ = ["get_image_blending"]
