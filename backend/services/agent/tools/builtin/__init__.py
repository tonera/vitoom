"""Agent 可挂载的内置工具集合。

目录约定：每个工具放在独立的模块中，使用 `@register_tool` 注册；
运行时由 `tool_plugin_registry` 自动扫描本目录下的子模块。
"""
