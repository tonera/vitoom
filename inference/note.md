# TurboDiffusion / SpargeAttn 重编

PyTorch 2.13 起删掉了 `c10::impl::cow::materialize_cow_storage`。升到 torch 2.14 后，旧扩展 import 会失败：

```text
undefined symbol: _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE
```

按**当前环境里的 torch** 打 wheel，再 `pip install`。不要从 PyPI 装，也不要手拷 `.so`。wheel 和编译它的 Python、CUDA、torch 版本绑在一起，换了 torch 就要重打。

两个包的 `setup.py` 已改成 C++20（torch 2.14 头文件要求）。SpargeAttn 不要加 `-Xcompiler -include,cassert`，GCC 13 会报 `std::__terminate` 重定义。

```bash
export PATH="/home/tonera/miniconda3/envs/vitoom-infer/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=6
```

`PATH` 里要有这个环境的 `ninja`，否则会退回 distutils。`--no-build-isolation` 让扩展链到已安装的 torch；`--no-deps` 避免把 torch 重新装一遍。版本号没变时必须加 `--force-reinstall`，否则 pip 认为已满足、不会换掉旧扩展。

## turbodiffusion

源码：`/home/tonera/project/TurboDiffusion`。架构在 `setup.py` 的 `cc_flag` 里：5090 用 `sm_120a`，GB10 用 `sm_121a`。`sm_120` 的 cubin 不能在 GB10 上跑。

```bash
cd /home/tonera/project/TurboDiffusion
pip wheel . --no-build-isolation --no-deps -w dist_wheels
pip install --force-reinstall --no-deps dist_wheels/turbodiffusion-*.whl
```

装进去的是 `turbo_diffusion_ops` 和一份 `turbodiffusion` Python 包。推理进程把 `inference/` 放在 `sys.path` 最前，实际跑的仍是仓库里的 `inference/turbodiffusion`。

```bash
cd /tmp
python -c "import torch; from turbo_diffusion_ops import quant_cuda, gemm_cuda; print('ok')"
```

先 `import torch`。扩展没有把 `torch/lib` 写进 RPATH，单独 import 会找不到 `libc10.so`。

## spas_sage_attn

源码：`/home/tonera/project/SpargeAttn`。`setup.py` 的 `SUPPORTED_ARCHS` 决定 `TORCH_CUDA_ARCH_LIST` 能不能用，不在名单里会在 nvcc 之前直接报错。

| 机器 | 计算能力 | `TORCH_CUDA_ARCH_LIST` |
| --- | --- | --- |
| RTX 5090 | 12.0（sm_120） | `8.6;8.9;12.0` |
| GB10 | 12.1（sm_121） | `12.1` |

5090 带上 `8.6` 会关掉 Sage2++，日志里的 `Sage2++ NOT enabled` 是预期现象。GB10 只编 `12.1` 时 Sage2++ 保持打开。大约 200 个实例化文件，编译要几分钟。

```bash
cd /home/tonera/project/SpargeAttn
# 5090: export TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0"
# GB10: export TORCH_CUDA_ARCH_LIST="12.1"
pip wheel . --no-build-isolation --no-deps -w dist_wheels
pip install --force-reinstall --no-deps dist_wheels/spas_sage_attn-*.whl
```

```bash
cd /tmp
python - <<'PY'
import sys
sys.path.insert(0, "/home/tonera/project/vitoom/inference/turbodiffusion")
import torch
from SLA.core import SAGESLA_ENABLED, SageSparseLinearAttention
assert SAGESLA_ENABLED
SageSparseLinearAttention(head_dim=128, topk=0.1)
print("ok")
PY
```

## 重启推理进程

`SAGESLA_ENABLED` 在第一次 import 时就定下来了。换过包之后重启视频进程，再发任务。

```bash
supervisorctl restart aigc_video_local_5090
```
