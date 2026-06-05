"""
mini 服务的组合 pipeline（层次：bridge < pipeline < handler）

- pipeline 不拥有模型生命周期（由 handler + bundle_cache 负责）
- pipeline 只负责把多个 bridge 的原子能力组合成一次可交付的产物
- 当前包含：doc_pipeline（GLM-OCR + doclayout-yolo 的图文混排 markdown 打包）
"""
