# LiveTalking 上游源码（参考用，**不在 vitoom 运行时链路中**）

## 目录定位

本目录原样保留了 [lipku/LiveTalking](https://github.com/lipku/livetalking) 的源代码快照，**仅作为对照参考**：

- vitoom 实际运行的 MuseTalk 推理内核已经 vendor 到 [`inference/avatar/livetalking/musetalk/`](../../avatar/livetalking/musetalk/)（精简后的 `vae.py` / `unet.py` / `audio2feature.py` / `blending.py` 等）
- vitoom 后端 / sidecar / 单测 **完全不 import 本目录下任何文件**（用 `sys.meta_path` blocker 校验过）
- 本目录的存在仅有两个用途：
  1. vendor 推理内核时对照源码 / 排查上游行为差异
  2. 等端到端联调通过、不再需要参考时，**整个目录可一刀删除**

## 不要做的事

> 以下行为已在历史中尝试过并废弃，请不要重新引入。

- ❌ **不要**修改本目录任何文件去对接 vitoom 的资源路径（如 `resources/models/livetalking/`）。
  历史上曾有 `vitoom_paths.py` + 6 处 `# vitoom-patch:` 改路径的方案，已撤销。
  vitoom 的所有路径常量统一放在 [`inference/avatar/livetalking/musetalk/paths.py`](../../avatar/livetalking/musetalk/paths.py)。
- ❌ **不要**在 vitoom 后端或 sidecar 里 `sys.path.insert` 把本目录加进去然后 `from avatars.musetalk_avatar import MuseReal`。
  上游 `BaseAvatar` 强制依赖 TTS plugin + WebRTC HumanPlayer，不适合当库类用，详见
  [`.cursor/plans/livetalking_装饰接入_*.plan.md`](../../../.cursor/plans/) 的"关键决策"小节。
- ❌ **不要**把本目录加进 `requirements.txt` 或当成 pip 包安装。LiveTalking 上游不发布 pip 包，
  我们走的是 vendor + 改写关键模块（`MuseTalkRuntime` / `FeatureBuffer`）的路线。

## 升级上游版本

如果未来需要 cherry-pick 上游修复或升级 MuseTalk 模型版本，正确流程是：

1. `git clone` 新版 LiveTalking 到本目录（覆盖）
2. **重新** vendor 到 `inference/avatar/livetalking/musetalk/`，diff 旧版找改动
3. 跑 `test/test_livetalking.py` 全套单测 + 端到端验证
4. **不要**给本目录任何文件加 patch 或新增文件
