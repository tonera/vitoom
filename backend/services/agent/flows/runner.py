from __future__ import annotations

import asyncio
from typing import Any, Dict


class FlowRunner:
    """第一阶段的最小执行包装层。"""

    async def run(self, crew: Any, *, inputs: Dict[str, Any]) -> Any:
        return await asyncio.to_thread(crew.kickoff, inputs=inputs)
