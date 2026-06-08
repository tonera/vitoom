"""数字人副链路相关后端组件。

模块职责（详见 .cursor/plans/livetalking_装饰接入_*.plan.md）：

* ``livetalking_config`` —— 从 ``inference_services`` 表读取 sidecar 注册状态
  （sidecar 启动时会调 ``POST /api/inference/services/{id}/start`` 上报
  ``host`` / ``port`` / ``config``）；带 TTL=2s 进程内缓存避免热路径每次查 DB
* ``livetalking_client`` —— 后端 → sidecar 的非阻塞 PCM 推流 client
  （resample + bounded queue + 独立 consumer task）
"""
