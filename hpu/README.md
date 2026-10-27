# hpu/ — Strata 的 Intel Gaudi2 原生后端

构建、运行和性能数据见仓库根目录的 [README](../README.md)。

| 目录 | 内容 |
|---|---|
| `src/`, `include/strata/hpu/` | 引擎：模型加载、各层图（GDN、QSA、MoE、HC、PLE）、DFlash verify 链、host 侧调度 |
| `tpc/` | 自写 TPC kernel（`glue.cpp` 是 kernel 注册表和形状检查） |
| `tests/` | 单元测试（`ctest`，大多不需要占卡），以及几个单 kernel 的计时 probe |
| `tools/` | 模型和设备信息工具 |
| `deploy/` | 生产配置和启停脚本 |
| `cmake/` | SynapseAI、GGUF（ggml）等依赖的查找 |

性能开关都在 `include/strata/hpu/performance_modes.hpp` 里，每个开关上方有注释说明；`deploy/strata-hpu-production.json` 是实际在用的组合。
