"""需要运行时动态生成配置的预置 Agent 集合。

目录约定：一个预置一个模块，在模块顶部用 `@register_preset` 注册，
运行时由 `preset_plugin_registry` 自动扫描本目录下的子模块。
"""
