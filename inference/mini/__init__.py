"""
Mini 推理服务

设计定位：
- 承载轻量、多种类、按需加载的"小模型工具集"（OCR / rerank / embed / layout …）
- 核心机制：LRU=1 + TTL 超时驱逐，一个常驻进程里按 job_type 切换不同 handler
- 对上协议：走 /v1/tasks（task_type="mini"），和 image/video/audio/text 一致

首发 handler：
- OCR（GLM-OCR，vLLM runtime）

扩展方式：
- 在 handlers/ 下新增 xxx_handler.py 实现 handle(params) 接口
- 在 MiniInferrer._HANDLERS 里按 job_type 注册
- 专属参数走 InferenceRequestParams.extract 字段承载
"""
