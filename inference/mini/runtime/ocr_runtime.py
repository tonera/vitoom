"""
OCR runtime 统一抽象

为什么存在：
- mini 服务的 OCR handler 同时支持两种后端：
    * transformers（AutoProcessor + AutoModelForImageTextToText；冷启动快、实现简单，默认）
    * vLLM（高吞吐、显存可控；冷启动慢，适合长期常驻高并发）
- 上层（OcrHandler / doc_pipeline）不希望分支 if/elif runtime，所以统一一个接口。

接口约定：
    bundle.generate_from_messages(messages, *, max_new_tokens=None, temperature=None, top_p=None) -> str
    bundle.shutdown() -> None

其中 messages 仍遵循 "MIT 风格"的多模态 chat：
    [{"role":"user","content":[
        {"type":"image","image": <path|PIL|URL>},
        {"type":"text","text": "Text Recognition:"},
    ]}]

两个后端各自在自己的 bridge 模块里实现这套接口（HfOcrBundle / VllmOcrBundle）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class OcrBundleLike(Protocol):
    """所有 OCR 后端 bundle 需要实现的最小接口。"""

    def generate_from_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_new_tokens: Any = None,
        temperature: Any = None,
        top_p: Any = None,
    ) -> str:
        ...

    def shutdown(self) -> None:
        ...


__all__ = ["OcrBundleLike"]
