"""
图片推理器模块
"""
from __future__ import annotations

"""
注意：不要在 package import 阶段引入 heavy/有副作用的模块（例如 inferrer），否则容易造成循环导入，
并让 pytest 在收集阶段就失败。

但为了兼容历史用法（from image import ImageInferrer），这里提供惰性导入。
"""

from typing import Any

__all__ = ["ImageInferrer"]


def __getattr__(name: str) -> Any:
    if name == "ImageInferrer":
        from .inferrer import ImageInferrer

        return ImageInferrer
    raise AttributeError(name)

