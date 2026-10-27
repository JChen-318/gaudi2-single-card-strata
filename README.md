# Strata on a single Intel Gaudi2

在**一张 Intel Gaudi2（HL-225，96 GB HBM）**上原生运行 Qwen3.8-Flash-Next（512 专家 MoE，48 层，256k 上下文）的推理引擎，
提供 OpenAI / Anthropic 兼容 API。

本仓库基于 [Strata](docs/STRATA_UPSTREAM_README.md)（MIT 许可，见 [LICENSE](LICENSE)）。在它的服务端（`serve/`）之下新增了一个
Gaudi2 原生后端 `hpu/`：模型的每一层都编译成 SynapseAI 图，在 MME 和 TPC 上执行，并配有一百多个自写的 TPC kernel。
上游的 CUDA/HIP/SYCL/Metal 后端原样保留，但与本后端无关。

## 实测性能

所有数字均在下文"测试环境"一节的机器上测得，模型为 IQ4_XS GGUF，单请求。

**decode**（DFlash2 投机解码，block 4，greedy，生产服务，[llm_speedtest](https://github.com/gengchaogit/llm_speedtest)，2026-10-26）：

| 上下文 | 2k | 8k | 32k | 64k | 128k |
|---|---|---|---|---|---|
| tok/s | 155.7 | 103.1 | 112.1 | 92.9 | 98.6 |

- 平均每轮 verify 接受约 2.9 个 token。2026-10-27 起又启用了 `GDN_STEP_ROWS` 和 `GEMV_PARTS_GU_WINDOW`：4 行 verify 窗口在 2k 上下文下从 19.26 ms 降到 18.63 ms（119 个窗口的截尾平均）。
- 不开 DFlash 时，单 token decode 约 12.8 ms/token（2k 上下文，probe）。

**prefill**（8192 行窗口）：

| prompt 长度 | 16k | 32k | 64k | 128k | 254k |
|---|---|---|---|---|---|
| tok/s | 3562 | 3264 | ~3112 | 2822 | 2404 |

16k 和 32k 是生产服务的实测；64k 及以上是单进程 probe 的实测，长上下文使用稀疏注意力。

## 测试环境

| 项目 | 版本 |
|---|---|
| 加速卡 | 1 × Intel Gaudi2 HL-225，98304 MiB HBM |
| 驱动 / SynapseAI | habanalabs 1.24.1 |
| TPC 编译器 | tpc-clang 2.7.33 |
| 系统 | Ubuntu 24.04.4，g++ 13.3，CMake 4.4，Python 3.12 |
| CPU / 内存 | Xeon E5-2696 v3，251 GB（PLE 表约 28.8 GB 放在内存中，建议 ≥ 128 GB） |

## 模型文件

- **目标模型**：[orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF) 中的 IQ4_XS 版本（3 个分片，config 中写第一个分片的路径）。
  - 512 个专家全部以 IQ4 码 + F32 scale 的形式常驻 HBM，约 76 GB。
  - 运行参数需要 `--hbm-limit-mib 94208 --vram-reserve-mib 2048`。
- **DFlash2 draft 模型**：[incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)（方法见 [z-lab/dflash](https://github.com/z-lab/dflash)），下载整个仓库目录，用环境变量 `STRATA_HPU_DFLASH` 指向它。
- **tokenizer 目录**：Strata setup 为该模型生成的 tokenizer（`pack/full/tokenizer`，参见 [docs/AI_SETUP.md](docs/AI_SETUP.md)），用 config 的 `"tokenizer"` 字段指向它。

## Docker（推荐）

[`hpu/docker/`](hpu/docker/) 提供镜像和 compose 文件：

- 基础镜像是 Habana 1.24.1 官方镜像（Ubuntu 24.04），TPC 编译器 `tpc-clang` 从 Habana 的 apt 源安装；
- 引擎和 TPC kernel 在 `docker build` 时编译；
- 宿主机只需要装 habanalabs 驱动 1.24.x，不需要专门的容器运行时；
- 卡通过 `/dev/accel/accel0`、`/dev/accel/accel_controlD0` 映射进容器。
- 还要映射 `/dev/infiniband/uverbs0`，否则 SynapseAI 拿卡时会失败，报 `synDeviceAcquireByDeviceType ... status 26`。
- 首次启动要从磁盘读入约 100 GB 的模型和 PLE 表；在作者的 NAS 上，加上自动预热，大约 10 分钟后才能服务。

```bash
git clone https://github.com/JChen-318/gaudi2-single-card-strata.git
cd gaudi2-single-card-strata
# MODELS 目录下应有 qwen3.8-flash-next-uncensored-iq4xs/（GGUF 三个分片）和 qwen3.8-flash-next-dflash2/
MODELS=/path/to/models TOKENIZER=/path/to/tokenizer \
  docker compose -f hpu/docker/compose.yaml up -d --build
docker logs -f strata-gaudi2        # 等到出现 ready 和 warm-up done
```

或者不用 compose：

```bash
docker build -f hpu/docker/Dockerfile -t strata-gaudi2 .
docker run -d --name strata-gaudi2 --restart unless-stopped \
  --device /dev/accel/accel0 --device /dev/accel/accel_controlD0 --device /dev/infiniband/uverbs0 \
  --ipc host --network host --cap-add SYS_NICE --ulimit memlock=-1:-1 \
  -v /path/to/models:/models:ro -v /path/to/tokenizer:/tokenizer:ro \
  strata-gaudi2
```

容器支持的环境变量：

| 变量 | 作用 |
|---|---|
| `MODEL_GGUF`、`DFLASH_DIR`、`TOKENIZER_DIR` | 模型、drafter、tokenizer 在容器内的路径（默认值对应上面的挂载方式） |
| `STRATA_PORT` | 端口，默认 8080 |
| `API_KEY` | 设置后客户端必须带这个 key |
| `WARMUP` | 默认 1：服务就绪后用 8191、4095、2047 token 的 prompt 各跑一次，把常用窗口尺寸的图预先编译好 |
| `--build-arg PIP_INDEX_URL=...` | 构建时换 pip 镜像源 |

同一时间只能有一个进程占用这张卡，启动容器前请先停掉宿主机上的其他服务。

## 构建

```bash
git clone https://github.com/JChen-318/gaudi2-single-card-strata.git
cd gaudi2-single-card-strata
pip install -r requirements.txt
cmake -S hpu -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j8
```

- 需要已安装 SynapseAI（`/usr/include/habanalabs`、`libSynapse.so`、`libtpc_kernels.so`）和 `tpc-clang`。
- CMake 会下载固定版本的 llama.cpp（`3cf03257`），只用它的 ggml 来读 GGUF。离线环境可以改用 `-DSTRATA_HPU_GGML_DIR=<该版本的检出目录>`。
- `build-release/strata-hpu` 是引擎本体，`libstrata_hpu_tpc.so` 里是全部 TPC kernel。
- 单元测试：在构建目录下运行 `ctest`，不需要占卡。需要上卡的测试要加 `-DSTRATA_HPU_HARDWARE_TESTS=ON`。

## 运行

1. 编辑 [`hpu/deploy/strata-hpu-production.json`](hpu/deploy/strata-hpu-production.json)，把里面的路径改成本机实际位置：
   - `exe`：引擎可执行文件；
   - `cwd`：本仓库的检出目录；
   - `--native` 参数：GGUF 第一个分片；
   - `STRATA_HPU_DFLASH`：draft 模型目录；
   - `tokenizer`：tokenizer 目录；
   - `log`：引擎日志文件。

   这份就是作者生产服务在用的配置，`env` 里的 `STRATA_HPU_*` 开关决定启用哪些 kernel 和路径。每个开关的说明见 [`hpu/include/strata/hpu/performance_modes.hpp`](hpu/include/strata/hpu/performance_modes.hpp)。
2. 启动和停止：

   ```bash
   CONFIG=$PWD/hpu/deploy/strata-hpu-production.json PORT=8080 bash hpu/deploy/run-production.sh
   bash hpu/deploy/stop-production.sh
   ```

   引擎加载完成、报 READY 之后，API 地址是 `http://<host>:8080/v1`（例如 `/v1/chat/completions`），浏览器打开 `http://<host>:8080/` 即可聊天。
3. 默认没有 API key。如果服务暴露在本机以外，请在 config 里加上 `"api_key": "..."`。

## 主要实现

- **MoE**
  - decode 时，每个 token 选中的 10 个路由专家和 1 个共享专家，在 TPC 上直接用 IQ4×Q8 的 GEMV 计算（`hpu/tpc/iq4q8_gemv_packed.c`）。
  - prefill 时，先按专家把 IQ4 展开成 BF16 分块，再交给 MME 做分组 GEMM（`STRATA_HPU_MOE_MME`）。
- **稠密投影**：BF16 合并 GEMM，在 MME 上执行。
- **GDN（线性注意力层，36 层）**
  - conv、prepare、递推、norm 都是 TPC kernel；
  - prefill 用分块（chunked）GDN；
  - DFlash 的 4 行 verify 窗口用一个 kernel 完成 4 步递推。
- **QSA（全注意力层，每 4 层一层，共 12 层）**
  - FP8 KV cache；
  - 在设备上完成块级 top-k 选择；
  - 长上下文 prefill 用稀疏 gather 注意力，并用 Synapse 的 SDPA（把 GQA 折叠进行维）；
  - decode 和 verify 按选中的块做 gather 注意力。
- **DFlash2 投机解码**
  - 4 行 verify 窗口，argmax 在设备上计算；
  - 采样请求也能用，需要 `--spec-prob`。
- **HC 多流残差、PLE**：用融合的 TPC kernel，每层的图串成一条 launch 链，共用一个 workspace。

## 已知限制

- **单卡串行**：同时来多个请求时会排队，一个请求做 prefill 时，另一个请求的 decode 会暂停。
- **首次编译**：图按窗口大小和上下文分档编译，第一次遇到新的 prompt 尾窗口大小时要现场编译，长 prompt 首次可能多花 10 s 以上。
  - 启动后用长度为 8191、4095、2047 token 的 prompt 各请求一次，可以把常用尺寸预先编译好。
  - decode 跨过上下文分档线（32k 以内每 4096 token 一档）时，会有一次约 0.2 s 的停顿。
- 只支持 Qwen3.8-Flash-Next 这一模型结构，引擎会检查模型几何参数。

## 许可

MIT。上游 Strata 的版权声明见 [LICENSE](LICENSE)；`hpu/` 目录是本仓库新增的 Gaudi2 后端。
