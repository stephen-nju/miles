# 高性能 JSON 解析引擎削减 Session Server R3 解析开销

## 原始想法

---
title: Session Server Overhead
description: Benchmark evidence and optimization context for CPU overhead in the session-server TITO path.
---

# Session Server 开销

本说明记录了在使用 TITO 且请求 R3（`routed_experts`）元数据的多轮 agent loop 下，session-server CPU 开销的当前状态。它面向接下来要优化该 loop 的维护者；不是面向用户的调参指南。

## 动机

session server 的存在，是为了让 OpenAI 格式的多轮 agent loop 与 Miles 的 token-in/token-out 训练路径兼容。每个 chat 请求都会附着到一个 session 上，server 通过 TITO 构建 Miles 自有的 `input_ids`，把请求代理到后端，校验后端响应，存储该轮记录，之后让 rollout collection 从这些记录中重建样本。

棘手的不是普通文本聊天，而是启用了 R3、并在每次成功响应里携带巨大 all-token `routed_experts` 负载的 agent loop。在 `r3_scale=1000` 时，即便新的 user 和 assistant 内容各只有 1k tokens，一个长 session 的后段每轮响应体也可能超过 100 MiB。

## 范围与非目标

本文覆盖 TITO session 路径中可见的 Python session-server 开销：请求解析、TITO prompt 构建、注入 `input_ids` 的请求 dump、响应解析与校验，以及内存中记录的保留。它不覆盖 SGLang 模型执行、网络传输、Ray 调度、训练侧 loss 计算或真实生产端点延迟。

下面的数字是合成 CPU 基准。它们对判断方向和数量级有用，但不能替代 loop 优化实现后的完整端到端 rollout 基准。

## 当前实现

`SessionServer` 拥有一个有界的 CPU executor：

```python
self.cpu_executor = ThreadPoolExecutor(
    max_workers=getattr(args, "session_server_cpu_workers", None) or min(16, os.cpu_count() or 1),
    thread_name_prefix="session-cpu",
)
```

在当前机器上 `os.cpu_count()` 为 192，所以默认 `session_server_cpu_workers` 是 16。在当前 PR 中，请求 JSON 解析、请求 JSON dump 以及成功响应的解析/校验都被 offload 到该 executor。TITO tokenization 与状态变更仍在 event loop 上、于 per-session 锁下进行。

当前分支在追加记录时还会修剪旧的 all-token `routed_experts` / `indexer_topk` blob。这是内存保留优化，而非每轮解析优化：它只保留最后 rollback-depth + 1 条带大 blob 的记录，因为在 last-wins 合并之后，`merge_samples` 不会再读取更旧记录的 all-token blob。

## 基准设置

用于对比的基准参数为：

```bash
sessions=32
turns=50
input_tokens=1000
output_tokens=1000
r3_scale=1000
```

基准脚本位于 `tests/benchmark/bench_session_server_overhead.py`。该脚本生成合成的请求/响应体，并驱动真实的 `SessionRegistry` / `LinearTrajectory` TITO 路径以及响应校验 helper。

重要提醒：该脚本的默认运行是串行 CPU 微基准，不会实例化 `SessionServer.cpu_executor`。为测量 worker 行为，使用了一个临时 harness，把相同负载分成 50 波、每波 32 个并发 session turn 运行，并把请求解析、请求 dump、响应解析/校验提交给 `ThreadPoolExecutor(max_workers=16)`。

## 结果

| 运行 | 代码路径 | 墙钟时间 | 吞吐 | Reply p50 | Reply p95 | 响应解析 p50 | 响应解析 p95 | 保留 R3 raw 估计 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 当前串行微基准 | 当前分支，无 executor | 154.172s | 10.38 turns/s | 87.79ms | 193.58ms | 78.83ms | 176.53ms | 6084.7 MiB |
| 当前 16-worker executor harness | 当前分支，32 并发 session，`max_workers=16` | 192.889s | 8.3 turns/s | 3576.70ms | 7321.47ms | 3361.34ms | 6972.91ms | 6084.7 MiB |
| 旧 commit `1daf70714` | 旧 session 代码，串行兼容 harness | 176.475s | 9.07 turns/s | 108.81ms | 209.41ms | 101.50ms | 196.45ms | 78363.0 MiB |

这些运行的产物写到了：

- `/tmp/session-overhead-s32-t50-i1000-o1000-r3-1000.json`
- `/tmp/session-overhead-current-workers16-s32-t50-i1000-o1000-r3-1000.json`
- `/tmp/session-overhead-1daf70714-s32-t50-i1000-o1000-r3-1000.json`

在当前 16-worker 运行期间，`ps` 显示 RSS 约 22.5 GiB。在旧 `1daf70714` 运行期间，`ps` 显示 RSS 约 77.3 GiB。这些是观测到的某一时刻 RSS 读数，不是精确的峰值内存测量。

## 解读

主导成本是巨大 JSON 响应体的解析/校验，而非 TITO tokenization。在 16-worker 运行中，响应解析+校验 p50 为 3.36s、p95 为 6.97s，因为 32 个并发响应在争抢 16 个 worker 线程，且 Python 的 JSON 解析仍在 GIL 下运行。

executor offload 仍然有用，但它解决的是另一个问题。它把 CPU 密集的 JSON 工作移出 asyncio event loop，让 health check 等 event-loop 工作保持响应。它并不会让聚合 JSON 解析快多少，而且在高并发下会让单请求延迟看起来更糟，因为请求现在包含了 executor 排队时间。

旧 commit 的对比表明，修剪旧 R3 blob 对内存保留很重要。在相同合成 session 形态下，保留的 raw R3 从大约 76.5 GiB 降到大约 5.9 GiB。该改进并不消除解析最新那个巨大响应体的每轮成本，所以仅靠记录修剪，延迟和吞吐不会有大幅改善。

## 对 Loop 优化的启示

不要一上来就微优化 TITO tokenization。在所测负载中，即便在 16-worker 并发 harness 下，tokenization p50 也约为 5ms，而响应解析+校验 p50 是以秒计的。

不要指望增加 CPU worker 是干净的解法。更多 worker 也许能减少突发时的排队，但核心工作是对大字符串和大数组的 Python JSON 解析；GIL 与内存带宽仍会主导。更多 worker 还会在同时解析许多大响应体时加剧内存压力。

下一步优化应当减少 Python session loop 必须同时解析、拷贝、保留或调度的巨型 R3 负载量。候选方向有：在 trainer 能消费 delta 时，只返回每轮或 delta 的 R3 而非 all-token R3；让不透明的 R3 负载不进入 session server 必须完整 `json.loads` 的 JSON 对象；把 R3 移到二进制旁路通道或 artifact 引用；或改变 loop 调度器，使在飞的巨型响应解析数不超过机器能承受的量。

如果在改变负载格式之前先调 loop 级并发，应当把它当作遏制（containment）而非根因修复。把并发 session completion 上限压到 worker 数或更低，应能改善尾延迟，但会牺牲吞吐。除非修剪旧记录或负载本身变小，否则它也不会减少保留的内存。

## 失效模式与运维说明

若 `reply_latency_ms` 跟随 `response_parse_validate_ms`，说明 session server 在响应 JSON 处理上 CPU-bound。提高 `session_server_cpu_workers` 也许能移动曲线，但不太可能改变主要结论。

若 RSS 大致随 `sessions * turns * accumulated_tokens * r3_scale` 增长，说明旧记录在保留 all-token R3 blob。检查 `LinearTrajectory.append_record`；当前分支会有意地从比 `MAX_ASSISTANT_ROLLBACK_STEPS + 1` 更旧的记录中丢弃 `routed_experts` 和 `indexer_topk`。

若 event-loop 响应性看起来很差而总吞吐相近，使用 `tests/benchmark/bench_session_responsiveness.py`。该基准专门关注在 offload CPU 工作时保持 `/health` 响应，而非提升总 CPU 吞吐。

若基准结果看起来不一致，先确认测量的是哪条路径。默认的 overhead 基准是串行的、不使用 `SessionServer.cpu_executor`；评估 `session_server_cpu_workers` 需要 executor 或 HTTP 基准。

## 代码锚点

- `miles/rollout/session/session_server.py`：`SessionServer.__init__` 拥有 `cpu_executor` 与默认 worker 数。
- `miles/rollout/session/sessions.py`：`chat_completions`、`_parse_request_body`、`_dump_request_body`、`_parse_and_validate_response` 定义了请求/响应的 CPU 阶段以及 executor offload 发生的位置。
- `miles/rollout/session/linear_trajectory.py`：`LinearTrajectory.prepare_pretokenized`、`update_pretokenized_state`、`append_record` 定义了 TITO 状态变更与 R3 记录保留。
- `tests/benchmark/bench_session_server_overhead.py`：session 层开销的串行 CPU 微基准。
- `tests/benchmark/bench_session_responsiveness.py`：针对 executor-offload 动机的 HTTP/event-loop 响应性基准。

## 开放问题

- SGLang 或 router 能否返回 per-turn delta 的 R3 而非 all-token blob，且不破坏训练侧重建？
- session server 能否把 `routed_experts` 作为不透明的 byte/string 负载存储或转发，而不必把它完整解析进 Python 响应 dict？
- 正确的 loop 优化到底是协议变更、调度/并发上限，还是两者兼有？
- 目标 agent loop 在真实生产中的响应体大小分布与并发 session 数是多少？

## 主方向：高性能解析引擎替换

### 理由

在同一份数据、同一条路径上把 stdlib `json` 换成更快的解析器并减少拷贝，角度纯粹是"解析实现速度"，不触碰协议 / 传输 / 并发模型——因此它是最贴合现有 repo 模式、落地面最小、置信度最高的本地改造，且能把 Alt-2 的"惰性不解析"作为子机制自然吸收。

### 方案概要

将 stdlib `json` 替换为更高性能的解析库（`orjson` / `msgspec`），并用 typed 解析模式把"解析 + 校验"合并为单遍，重点把巨大的 R3 字段从物化路径上摘掉：

1. **解析库替换**：`_parse_request_body()` 用 `orjson.loads()` / `msgspec.json.decode()` 替代 `json.loads()`；`_dump_request_body()` 用 `orjson.dumps()` 替代 `json.dumps()`；`_parse_and_validate_response()` 用 `msgspec` 的 typed decoding。
2. **Typed Schema 定义**：用 `msgspec.Struct`（或既有的 Pydantic `TypeAdapter`）描述 OpenAI chat completion response 结构，尤其是 `choices[0].meta_info.output_token_logprobs` 部分；把当前 `_parse_and_validate_response` 里手写的逐元素校验循环（`logprob` 必须是有限 `float`、`token_id` 必须是正整数且非 `bool`）转成 schema 内联的声明式校验，单遍完成解析与验证。
3. **关键优化（吸收 Alt-2）**：把 `routed_experts` / `indexer_topk` 声明为 `msgspec.Raw`（原始字节视图），让解析器跳过这两个巨型字段的物化——避免为 100+ MiB 的 base64 串分配 Python `str` / `list`。这把"惰性不解析"直接折叠进同一条解析路径。
4. **受影响组件**：`miles/rollout/session/sessions.py` 的 `_parse_request_body` / `_dump_request_body` / `_parse_and_validate_response`（核心热路径）；`miles/rollout/session/session_server.py` 的 error-body `json.dumps`（非热路径）；`tests/benchmark/bench_session_server_overhead.py` 的 JSON 操作需同步调整以反映新解析器。

### 客观证据

- `orjson` 3.11.9 与 `msgspec` 0.21.1 在当前环境中可直接 `import`（已验证 `python -c "import orjson, msgspec"` 成功），但二者均**未**在 `requirements.txt` / `pyproject.toml` 中声明——需补充为显式依赖。
- `examples/geo3k_vlm_multi_turn/env_geo3k.py` 已有 `orjson` 使用先例，采用可选导入 + fallback 到 stdlib `json` 的模式。
- Pydantic `TypeAdapter` 已是项目既有模式：`miles/rollout/generate_utils/tool_call_utils.py`、`miles/utils/chat_template_utils/template.py`、`miles/utils/test_utils/mock_sglang_server.py`。
- JSON 调用点明确且数量有限：`miles/rollout/session/sessions.py` 的 `_parse_request_body` / `_dump_request_body` / `_parse_and_validate_response` 为热路径，经 `ThreadPoolExecutor` offload；`session_server.py` 的 error-body `json.dumps` 为非关键路径。
- 校验逻辑可 schema 化：`_parse_and_validate_response` 当前的 type-checking 与 `math.isfinite()` 校验（近期 commit "reject non-finite logprobs" / "reject non-numeric logprob values"）可直接转译为 `msgspec` / Pydantic 声明式校验。
- `routed_experts` / `indexer_topk` 在响应里以 base64 字符串存放、下游 `get_routed_experts_from_response` / `_decode_topk_buffer`（`miles/rollout/generate_utils/generate_endpoint_utils.py`）期望字符串而非 list（见 `tests/benchmark/bench_session_server_overhead.py` 中 `"routed_experts": r3_blob`），故可用 `msgspec.Raw` 透传、无需物化。
- numpy 已广泛使用，`examples/random_async/random_async_rollout.py` 的 `_decode_routed_experts` 用 `np.frombuffer()` 从 bytes 直接得到 typed 数组，可与 `Raw` 配合做延迟解码。
- 基准 `tests/benchmark/bench_session_server_overhead.py` 量化了 `response_parse_validate` 为主要瓶颈（idea note 中 p50 3.36s / p95 6.97s）。
- 修改面可控：核心约 70 LOC（三个函数 + 一个 schema 定义），60 行命令式校验循环可压成约 20 行声明式 schema。

### 已知风险

- **Breaking change 风险**：`orjson` / `msgspec` 对 `NaN` / `Infinity` 等边界常量的处理与 stdlib `json` 略有差异；需验证上游 SGLang response 格式不会被新 parser 拒绝。
- **依赖声明风险**：`orjson` / `msgspec` 当前未在 `requirements.txt` 声明；需补依赖并处理最小化安装场景（可参考 `env_geo3k.py` 的可选导入 fallback）。
- **Schema 维护成本**：若 SGLang response schema 升级，需同步更新 `msgspec` / Pydantic schema；现有后验校验相对更灵活。
- **线程安全 / 零拷贝边界**：`orjson` / `msgspec` 本身线程安全，但 `ThreadPoolExecutor` 下需确保 `msgspec.Raw` 的 zero-copy 视图不会跨线程长期持有底层 buffer。

## 备选方向

### Alt-1：R3 增量协议下发
- 概要：从数据源头改协议——SGLang / router 在 `return_routed_experts` 时只回传本轮新增 tokens 的 per-turn delta（例如 `routed_experts_delta` + `routed_experts_prev_len`），而非累积的 all-token blob；`SessionRecord` 单条仅含 O(本轮 tokens) 的 R3，样本重建在 `compute_samples_from_openai_records` 中按序拼接 delta 还原完整 per-sample 矩阵，训练侧 `Sample.rollout_routed_experts` 契约不变。
- 客观证据：
  - `miles/rollout/session/sessions.py` 的 `_parse_and_validate_response` 当前从 `meta_info` 提取 `routed_experts`。
  - `miles/rollout/session/linear_trajectory.py` 的 `append_record` 已对旧 record 做 O(prefix) 剪枝（`MAX_ASSISTANT_ROLLBACK_STEPS = 1`，仅保留最后 2 条带大 blob 的 record），delta 模式下此剪枝变冗余。
  - `miles/rollout/generate_utils/openai_endpoint_utils.py` 的 `compute_samples_from_openai_records` 是新增 delta 累积逻辑的自然位置。
  - `miles/rollout/generate_utils/sample_utils.py` 的 `_merge_sample_pair` 对 `rollout_routed_experts` 采用 last-wins。
  - `miles/utils/types.py` 的 `Sample.rollout_routed_experts` 为 `numpy.ndarray`，是训练侧契约；`miles/ray/rollout/train_data_conversion.py` 原样转发，无需改动。
- 为何不作主方向：需要改 SGLang / router 的返回契约（跨仓库、需版本门控），SGLang 侧在本仓库不可见且协调成本高，置信度 medium，落地面与外部依赖均大于纯本地的解析引擎替换。

### Alt-2：R3 payload 惰性不解析
- 概要：不改协议也不改数据量，让 session 路径把 `routed_experts` / `indexer_topk` 当作不透明原始切片透传、永不 `json.loads` 成 Python list；`_parse_and_validate_response` 只解析校验真正需要的字段（`output_token_logprobs` / `completion_tokens` / `message.content`），R3 以原始字符串 / 字节存入 record，`append_record` 的 `.pop()` 对字符串键仍可用，下游解码逻辑不变。
- 客观证据：
  - `_parse_and_validate_response` 仅访问 `choices[0]` / `meta_info` / `output_token_logprobs` / `completion_tokens` / `message.content`，从不访问 `routed_experts` / `indexer_topk`。
  - `linear_trajectory.py` 的 `append_record` 仅用 `.pop()` 删除旧 R3，不需要解析其值。
  - `generate_endpoint_utils.py` 的 `get_routed_experts_from_response` / `_decode_topk_buffer` 期望 base64 字符串。
  - `bench_session_server_overhead.py` 中响应已以 `"routed_experts": r3_blob`（base64 字符串）形式存放。
- 为何不作主方向：用 stdlib 实现"只切出某字段原始值而不全量解析"需自写部分解析器，截断 / 转义等边界处理脆弱；其最干净的实现恰是 `msgspec.Raw`——这已被 Primary 吸收，故更适合作为 Primary 的核心子机制而非独立方向。

### Alt-3：R3 二进制旁路通道
- 概要：把 R3 整体移出 HTTP JSON 响应体——后端用 `ray.put()` 写入 Ray object store，响应仅回传一个 `ObjectRef` 句柄；`SessionRecord` 只存句柄（约数十字节），样本收集阶段用 `ray.get()` 延迟解引用；session server 不再需要 `json.loads` 100+ MiB 的 blob。
- 客观证据：
  - `miles/ray/rollout/train_data_conversion.py` 已用 Ray object store 传大训练数据（`Box(ray.put(rollout_data))`），且 `rollout_data` 含 `rollout_routed_experts`，是同一条数据流。
  - `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py` 已 `import` `ObjectRef` 用于权重同步，证明 ObjectRef 句柄作为跨进程引用可行。
  - `generate_endpoint_utils.py` 的 `return_routed_experts` 标志已存在，可扩展为 ObjectRef 传输开关。
- 为何不作主方向：会话服务器当前是独立 HTTP 代理、可不依赖 Ray；引入 object store 会带来集群依赖、对象生命周期 / eviction 管理与跨节点序列化等部署复杂度，落地面与风险大于纯本地的解析改造。

### Alt-4：并发准入与背压
- 概要：不改协议 / 解析，在 `SessionServer` 增加一个全局 `asyncio.Semaphore`（默认 = `session_server_cpu_workers`，可经新 flag 配置），在 `chat_completions` 提交 `_parse_and_validate_response` 前用 `async with` 门控，限制同时在飞的大响应解析数以削峰 p95；可选再叠加 size-aware 字节预算。
- 客观证据：
  - `session_server.py` 的 `__init__` 拥有 `ThreadPoolExecutor(min(16, os.cpu_count()))`，但无并发准入门。
  - `sessions.py` 在 `run_in_executor` 提交 parse；已有 per-session `chat_inflight` 门（每会话 1 并发）但无全局背压。
  - 既有 semaphore 先例：`sglang_rollout.py` 的 `GenerateState` semaphore、`inference_rollout_common.py` 的 `generate_fn_semaphore`。
- 为何不作主方向：idea note 明确把并发上限定位为 "containment rather than a root fix"——改善尾延迟但牺牲吞吐、不减内存、不降单次解析成本；更适合作为 Primary 的补充防护层而非主方向（置信度 high，但定位非根因）。

### Alt-5：进程级并行绕过 GIL
- 概要：把 `_parse_and_validate_response` 从受 GIL 约束的 `ThreadPoolExecutor` 换到 `ProcessPoolExecutor`（或 Ray actor），用多进程实现真正的多核并行解析；为压低 IPC，可只在 worker 内解析并返回小摘要，或用 `multiprocessing.shared_memory` / `/dev/shm` 传 100 MiB 响应字节。
- 客观证据：
  - 已有多进程先例：`examples/formal_math/single_round/prepare_data.py` 的 `ProcessPoolExecutor(max_workers=64)`；`router_manager.py` 用 `multiprocessing.Process`。
  - `_parse_and_validate_response` 文档注明 "Touches no session state"，纯 CPU、返回值可序列化，天然适合进程隔离。
  - session 状态变更在锁内、解析 / 转储在锁外（`sessions.py`），不阻碍进程化。
- 为何不作主方向：100+ MiB 响应体每次跨进程 pickle 往返的 IPC 成本很可能抵消多核收益；PEP 734 子解释器需 Python 3.13+ 而 `setup.py` 要求 `python_requires=">=3.10"`；净收益与置信度均不及解析引擎替换。

## 综合说明

若用户改选其它方向，有几条可与 Primary 叠加或前后接力：Alt-2「惰性不解析」本质就是 Primary 的核心子机制（`msgspec.Raw` 透传 `routed_experts` / `indexer_topk`），二者是同一条解析路径，无需当作互斥选项；Alt-4「并发准入」与 Primary 正交且互补，可在解析提速后作为防护层进一步压 p95 与峰值内存。Alt-1「增量协议」与 Alt-3「旁路通道」从源头削减 / 转移 payload，收益上限最高，适合作为协议侧的第二阶段——一旦 SGLang 契约可改，delta 或 ObjectRef 能把"根本不传 / 不解析大 blob"做到极致，届时本地解析提速退化为兜底层。Alt-5 则在 Python 升到 3.13+、或先 profiling 证明 IPC 收益为正后，可作为解析的并行执行后端替换 `ThreadPoolExecutor`。建议落地顺序：先 Primary（含 Alt-2 的 `Raw` 透传）拿到确定的本地收益，叠 Alt-4 控尾延迟，再评估 Alt-1 / Alt-3 的协议侧改造。
