# 高性能 JSON 解析引擎削减 Session Server R3 解析开销

## Original Idea

---
title: Session Server Overhead
description: Benchmark evidence and optimization context for CPU overhead in the session-server TITO path.
---

# Session Server Overhead

This note records the current state of session-server CPU overhead for multi-turn agent loops that use TITO and request R3 (`routed_experts`) metadata. It is meant for maintainers who will optimize the loop next; it is not a user-facing tuning guide.

## Motivation

The session server exists to make OpenAI-format multi-turn agent loops compatible with Miles's token-in/token-out training path. Each chat request is attached to a session, the server builds Miles-owned `input_ids` through TITO, proxies the request to the backend, validates the backend response, stores the turn record, and later lets rollout collection reconstruct samples from those records.

The painful case is not ordinary text chat. The painful case is an agent loop that enables R3 and carries a large all-token `routed_experts` payload in every successful response. At `r3_scale=1000`, a long session can make each late-turn response body exceed 100 MiB even when the new user and assistant content are only 1k tokens each.

## Scope and Non-goals

This doc covers the Python session-server overhead visible in the TITO session path: request parsing, TITO prompt construction, request dumping with injected `input_ids`, response parsing and validation, and in-memory record retention. It does not cover SGLang model execution, network transfer, Ray scheduling, trainer-side loss computation, or real production endpoint latency.

The numbers below are synthetic CPU benchmarks. They are useful for direction and order-of-magnitude decisions, but they are not a substitute for a full end-to-end rollout benchmark once the loop optimization is implemented.

## Current Implementation

`SessionServer` owns a bounded CPU executor:

```python
self.cpu_executor = ThreadPoolExecutor(
    max_workers=getattr(args, "session_server_cpu_workers", None) or min(16, os.cpu_count() or 1),
    thread_name_prefix="session-cpu",
)
```

On the current machine `os.cpu_count()` is 192, so the default `session_server_cpu_workers` is 16. In the current PR, request JSON parsing, request JSON dumping, and successful response parse/validation are offloaded to that executor. TITO tokenization and state mutation still happen on the event loop under the per-session lock.

The current branch also prunes old all-token `routed_experts` / `indexer_topk` blobs when appending records. That is a memory-retention optimization, not a per-turn parse optimization: it keeps only the last rollback-depth + 1 records with large blobs, because older records' all-token blobs are not read by `merge_samples` after the last-wins merge.

## Benchmark Setup

The benchmark parameters used for the comparison were:

```bash
sessions=32
turns=50
input_tokens=1000
output_tokens=1000
r3_scale=1000
```

The benchmark script lives at `tests/benchmark/bench_session_server_overhead.py`. That script generates synthetic request/response bodies and drives the real `SessionRegistry` / `LinearTrajectory` TITO path plus the response validation helper.

Important caveat: the script's default run is a serial CPU micro-benchmark and does not instantiate `SessionServer.cpu_executor`. To measure the worker behavior, a temporary harness was used that ran the same workload in 50 waves of 32 concurrent session turns, with request parse, request dump, and response parse/validate submitted to `ThreadPoolExecutor(max_workers=16)`.

## Results

| Run | Code path | Wall time | Throughput | Reply p50 | Reply p95 | Response parse p50 | Response parse p95 | Retained R3 raw estimate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| current serial micro-benchmark | current branch, no executor | 154.172s | 10.38 turns/s | 87.79ms | 193.58ms | 78.83ms | 176.53ms | 6084.7 MiB |
| current 16-worker executor harness | current branch, 32 concurrent sessions, `max_workers=16` | 192.889s | 8.3 turns/s | 3576.70ms | 7321.47ms | 3361.34ms | 6972.91ms | 6084.7 MiB |
| old commit `1daf70714` | old session code, serial-compatible harness | 176.475s | 9.07 turns/s | 108.81ms | 209.41ms | 101.50ms | 196.45ms | 78363.0 MiB |

Artifacts from these runs were written to:

- `/tmp/session-overhead-s32-t50-i1000-o1000-r3-1000.json`
- `/tmp/session-overhead-current-workers16-s32-t50-i1000-o1000-r3-1000.json`
- `/tmp/session-overhead-1daf70714-s32-t50-i1000-o1000-r3-1000.json`

During the current 16-worker run, `ps` showed RSS around 22.5 GiB. During the old `1daf70714` run, `ps` showed RSS around 77.3 GiB. These are observed point-in-time RSS readings, not exact peak memory measurements.

## Interpretation

The dominant cost is response parse/validation of huge JSON response bodies, not TITO tokenization. In the 16-worker run, response parse+validate p50 was 3.36s and p95 was 6.97s because 32 concurrent responses were competing for 16 worker threads and Python's JSON parsing still runs under the GIL.

The executor offload is still useful, but it solves a different problem. It keeps CPU-heavy JSON work off the asyncio event loop so health checks and other event-loop work can stay responsive. It does not make aggregate JSON parsing much faster, and it can make per-request latency look much worse under high concurrency because requests now include executor queueing time.

The old commit comparison shows that pruning old R3 blobs matters for memory retention. Under the same synthetic session shape, retained raw R3 dropped from roughly 76.5 GiB to roughly 5.9 GiB. That improvement does not remove the per-turn cost of parsing the latest huge response body, so latency and throughput do not improve dramatically from record pruning alone.

## Implications for Loop Optimization

Do not start by micro-optimizing TITO tokenization. In the measured workload, tokenization p50 was around 5ms even under the 16-worker concurrent harness, while response parse+validate p50 was measured in seconds.

Do not expect more CPU workers to be a clean fix. More workers may reduce queueing for bursts, but the core work is large Python JSON parsing of large strings and arrays; the GIL and memory bandwidth will still dominate. More workers can also increase simultaneous memory pressure when many large response bodies are parsed at once.

The next optimization should reduce how much giant R3 payload the Python session loop must parse, copy, retain, or schedule at the same time. Candidate directions are: return only per-turn or delta R3 instead of all-token R3 when the trainer can consume deltas; keep opaque R3 payloads out of the JSON object that the session server must fully `json.loads`; move R3 to a binary side channel or artifact reference; or change the loop scheduler so it does not allow more huge-response parses in flight than the machine can handle.

If loop-level concurrency is tuned before the payload format changes, treat it as containment rather than a root fix. Capping concurrent session completions at or below the worker count should improve tail latency but will trade away throughput. It also does not reduce retained memory unless old records are pruned or the payload itself is smaller.

## Failure Modes and Operational Notes

If `reply_latency_ms` tracks `response_parse_validate_ms`, the session server is CPU-bound on response JSON handling. Raising `session_server_cpu_workers` may shift the curve, but it is unlikely to change the main conclusion.

If RSS grows approximately with `sessions * turns * accumulated_tokens * r3_scale`, old records are retaining all-token R3 blobs. Check `LinearTrajectory.append_record`; the current branch intentionally drops `routed_experts` and `indexer_topk` from records older than `MAX_ASSISTANT_ROLLBACK_STEPS + 1`.

If event-loop responsiveness looks bad while total throughput is similar, use `tests/benchmark/bench_session_responsiveness.py`. That benchmark is specifically about keeping `/health` responsive while CPU work is offloaded, not about improving total CPU throughput.

If benchmark results look inconsistent, first check which path was measured. The default overhead benchmark is serial and does not use `SessionServer.cpu_executor`; an executor or HTTP benchmark is needed to evaluate `session_server_cpu_workers`.

## Code Anchors

- `miles/rollout/session/session_server.py`: `SessionServer.__init__` owns `cpu_executor` and the default worker count.
- `miles/rollout/session/sessions.py`: `chat_completions`, `_parse_request_body`, `_dump_request_body`, and `_parse_and_validate_response` define the request/response CPU stages and where executor offload occurs.
- `miles/rollout/session/linear_trajectory.py`: `LinearTrajectory.prepare_pretokenized`, `update_pretokenized_state`, and `append_record` define TITO state mutation and R3 record retention.
- `tests/benchmark/bench_session_server_overhead.py`: serial CPU micro-benchmark for session-layer overhead.
- `tests/benchmark/bench_session_responsiveness.py`: HTTP/event-loop responsiveness benchmark for the executor-offload motivation.

## Open Questions

- Can SGLang or the router return R3 as a per-turn delta instead of an all-token blob without breaking training reconstruction?
- Can the session server store or forward `routed_experts` as an opaque byte/string payload without fully parsing it into a Python response dict?
- Is the correct loop optimization a protocol change, a scheduler/concurrency cap, or both?
- What is the real production distribution of response body sizes and concurrent session count for the target agent loop?

## Primary Direction: 高性能解析引擎替换

### Rationale

在同一份数据、同一条路径上把 stdlib `json` 换成更快的解析器并减少拷贝，角度纯粹是"解析实现速度"，不触碰协议 / 传输 / 并发模型——因此它是最贴合现有 repo 模式、落地面最小、置信度最高的本地改造，且能把 Alt-2 的"惰性不解析"作为子机制自然吸收。

### Approach Summary

将 stdlib `json` 替换为更高性能的解析库（`orjson` / `msgspec`），并用 typed 解析模式把"解析 + 校验"合并为单遍，重点把巨大的 R3 字段从物化路径上摘掉：

1. **解析库替换**：`_parse_request_body()` 用 `orjson.loads()` / `msgspec.json.decode()` 替代 `json.loads()`；`_dump_request_body()` 用 `orjson.dumps()` 替代 `json.dumps()`；`_parse_and_validate_response()` 用 `msgspec` 的 typed decoding。
2. **Typed Schema 定义**：用 `msgspec.Struct`（或既有的 Pydantic `TypeAdapter`）描述 OpenAI chat completion response 结构，尤其是 `choices[0].meta_info.output_token_logprobs` 部分；把当前 `_parse_and_validate_response` 里手写的逐元素校验循环（`logprob` 必须是有限 `float`、`token_id` 必须是正整数且非 `bool`）转成 schema 内联的声明式校验，单遍完成解析与验证。
3. **关键优化（吸收 Alt-2）**：把 `routed_experts` / `indexer_topk` 声明为 `msgspec.Raw`（原始字节视图），让解析器跳过这两个巨型字段的物化——避免为 100+ MiB 的 base64 串分配 Python `str` / `list`。这把"惰性不解析"直接折叠进同一条解析路径。
4. **受影响组件**：`miles/rollout/session/sessions.py` 的 `_parse_request_body` / `_dump_request_body` / `_parse_and_validate_response`（核心热路径）；`miles/rollout/session/session_server.py` 的 error-body `json.dumps`（非热路径）；`tests/benchmark/bench_session_server_overhead.py` 的 JSON 操作需同步调整以反映新解析器。

### Objective Evidence

- `orjson` 3.11.9 与 `msgspec` 0.21.1 在当前环境中可直接 `import`（已验证 `python -c "import orjson, msgspec"` 成功），但二者均**未**在 `requirements.txt` / `pyproject.toml` 中声明——需补充为显式依赖。
- `examples/geo3k_vlm_multi_turn/env_geo3k.py` 已有 `orjson` 使用先例，采用可选导入 + fallback 到 stdlib `json` 的模式。
- Pydantic `TypeAdapter` 已是项目既有模式：`miles/rollout/generate_utils/tool_call_utils.py`、`miles/utils/chat_template_utils/template.py`、`miles/utils/test_utils/mock_sglang_server.py`。
- JSON 调用点明确且数量有限：`miles/rollout/session/sessions.py` 的 `_parse_request_body` / `_dump_request_body` / `_parse_and_validate_response` 为热路径，经 `ThreadPoolExecutor` offload；`session_server.py` 的 error-body `json.dumps` 为非关键路径。
- 校验逻辑可 schema 化：`_parse_and_validate_response` 当前的 type-checking 与 `math.isfinite()` 校验（近期 commit "reject non-finite logprobs" / "reject non-numeric logprob values"）可直接转译为 `msgspec` / Pydantic 声明式校验。
- `routed_experts` / `indexer_topk` 在响应里以 base64 字符串存放、下游 `get_routed_experts_from_response` / `_decode_topk_buffer`（`miles/rollout/generate_utils/generate_endpoint_utils.py`）期望字符串而非 list（见 `tests/benchmark/bench_session_server_overhead.py` 中 `"routed_experts": r3_blob`），故可用 `msgspec.Raw` 透传、无需物化。
- numpy 已广泛使用，`examples/random_async/random_async_rollout.py` 的 `_decode_routed_experts` 用 `np.frombuffer()` 从 bytes 直接得到 typed 数组，可与 `Raw` 配合做延迟解码。
- 基准 `tests/benchmark/bench_session_server_overhead.py` 量化了 `response_parse_validate` 为主要瓶颈（idea note 中 p50 3.36s / p95 6.97s）。
- 修改面可控：核心约 70 LOC（三个函数 + 一个 schema 定义），60 行命令式校验循环可压成约 20 行声明式 schema。

### Known Risks

- **Breaking change 风险**：`orjson` / `msgspec` 对 `NaN` / `Infinity` 等边界常量的处理与 stdlib `json` 略有差异；需验证上游 SGLang response 格式不会被新 parser 拒绝。
- **依赖声明风险**：`orjson` / `msgspec` 当前未在 `requirements.txt` 声明；需补依赖并处理最小化安装场景（可参考 `env_geo3k.py` 的可选导入 fallback）。
- **Schema 维护成本**：若 SGLang response schema 升级，需同步更新 `msgspec` / Pydantic schema；现有后验校验相对更灵活。
- **线程安全 / 零拷贝边界**：`orjson` / `msgspec` 本身线程安全，但 `ThreadPoolExecutor` 下需确保 `msgspec.Raw` 的 zero-copy 视图不会跨线程长期持有底层 buffer。

## Alternative Directions Considered

### Alt-1: R3 增量协议下发
- Gist: 从数据源头改协议——SGLang / router 在 `return_routed_experts` 时只回传本轮新增 tokens 的 per-turn delta（例如 `routed_experts_delta` + `routed_experts_prev_len`），而非累积的 all-token blob；`SessionRecord` 单条仅含 O(本轮 tokens) 的 R3，样本重建在 `compute_samples_from_openai_records` 中按序拼接 delta 还原完整 per-sample 矩阵，训练侧 `Sample.rollout_routed_experts` 契约不变。
- Objective Evidence:
  - `miles/rollout/session/sessions.py` 的 `_parse_and_validate_response` 当前从 `meta_info` 提取 `routed_experts`。
  - `miles/rollout/session/linear_trajectory.py` 的 `append_record` 已对旧 record 做 O(prefix) 剪枝（`MAX_ASSISTANT_ROLLBACK_STEPS = 1`，仅保留最后 2 条带大 blob 的 record），delta 模式下此剪枝变冗余。
  - `miles/rollout/generate_utils/openai_endpoint_utils.py` 的 `compute_samples_from_openai_records` 是新增 delta 累积逻辑的自然位置。
  - `miles/rollout/generate_utils/sample_utils.py` 的 `_merge_sample_pair` 对 `rollout_routed_experts` 采用 last-wins。
  - `miles/utils/types.py` 的 `Sample.rollout_routed_experts` 为 `numpy.ndarray`，是训练侧契约；`miles/ray/rollout/train_data_conversion.py` 原样转发，无需改动。
- Why not primary: 需要改 SGLang / router 的返回契约（跨仓库、需版本门控），SGLang 侧在本仓库不可见且协调成本高，置信度 medium，落地面与外部依赖均大于纯本地的解析引擎替换。

### Alt-2: R3 payload 惰性不解析
- Gist: 不改协议也不改数据量，让 session 路径把 `routed_experts` / `indexer_topk` 当作不透明原始切片透传、永不 `json.loads` 成 Python list；`_parse_and_validate_response` 只解析校验真正需要的字段（`output_token_logprobs` / `completion_tokens` / `message.content`），R3 以原始字符串 / 字节存入 record，`append_record` 的 `.pop()` 对字符串键仍可用，下游解码逻辑不变。
- Objective Evidence:
  - `_parse_and_validate_response` 仅访问 `choices[0]` / `meta_info` / `output_token_logprobs` / `completion_tokens` / `message.content`，从不访问 `routed_experts` / `indexer_topk`。
  - `linear_trajectory.py` 的 `append_record` 仅用 `.pop()` 删除旧 R3，不需要解析其值。
  - `generate_endpoint_utils.py` 的 `get_routed_experts_from_response` / `_decode_topk_buffer` 期望 base64 字符串。
  - `bench_session_server_overhead.py` 中响应已以 `"routed_experts": r3_blob`（base64 字符串）形式存放。
- Why not primary: 用 stdlib 实现"只切出某字段原始值而不全量解析"需自写部分解析器，截断 / 转义等边界处理脆弱；其最干净的实现恰是 `msgspec.Raw`——这已被 Primary 吸收，故更适合作为 Primary 的核心子机制而非独立方向。

### Alt-3: R3 二进制旁路通道
- Gist: 把 R3 整体移出 HTTP JSON 响应体——后端用 `ray.put()` 写入 Ray object store，响应仅回传一个 `ObjectRef` 句柄；`SessionRecord` 只存句柄（约数十字节），样本收集阶段用 `ray.get()` 延迟解引用；session server 不再需要 `json.loads` 100+ MiB 的 blob。
- Objective Evidence:
  - `miles/ray/rollout/train_data_conversion.py` 已用 Ray object store 传大训练数据（`Box(ray.put(rollout_data))`），且 `rollout_data` 含 `rollout_routed_experts`，是同一条数据流。
  - `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py` 已 `import` `ObjectRef` 用于权重同步，证明 ObjectRef 句柄作为跨进程引用可行。
  - `generate_endpoint_utils.py` 的 `return_routed_experts` 标志已存在，可扩展为 ObjectRef 传输开关。
- Why not primary: 会话服务器当前是独立 HTTP 代理、可不依赖 Ray；引入 object store 会带来集群依赖、对象生命周期 / eviction 管理与跨节点序列化等部署复杂度，落地面与风险大于纯本地的解析改造。

### Alt-4: 并发准入与背压
- Gist: 不改协议 / 解析，在 `SessionServer` 增加一个全局 `asyncio.Semaphore`（默认 = `session_server_cpu_workers`，可经新 flag 配置），在 `chat_completions` 提交 `_parse_and_validate_response` 前用 `async with` 门控，限制同时在飞的大响应解析数以削峰 p95；可选再叠加 size-aware 字节预算。
- Objective Evidence:
  - `session_server.py` 的 `__init__` 拥有 `ThreadPoolExecutor(min(16, os.cpu_count()))`，但无并发准入门。
  - `sessions.py` 在 `run_in_executor` 提交 parse；已有 per-session `chat_inflight` 门（每会话 1 并发）但无全局背压。
  - 既有 semaphore 先例：`sglang_rollout.py` 的 `GenerateState` semaphore、`inference_rollout_common.py` 的 `generate_fn_semaphore`。
- Why not primary: idea note 明确把并发上限定位为 "containment rather than a root fix"——改善尾延迟但牺牲吞吐、不减内存、不降单次解析成本；更适合作为 Primary 的补充防护层而非主方向（置信度 high，但定位非根因）。

### Alt-5: 进程级并行绕过 GIL
- Gist: 把 `_parse_and_validate_response` 从受 GIL 约束的 `ThreadPoolExecutor` 换到 `ProcessPoolExecutor`（或 Ray actor），用多进程实现真正的多核并行解析；为压低 IPC，可只在 worker 内解析并返回小摘要，或用 `multiprocessing.shared_memory` / `/dev/shm` 传 100 MiB 响应字节。
- Objective Evidence:
  - 已有多进程先例：`examples/formal_math/single_round/prepare_data.py` 的 `ProcessPoolExecutor(max_workers=64)`；`router_manager.py` 用 `multiprocessing.Process`。
  - `_parse_and_validate_response` 文档注明 "Touches no session state"，纯 CPU、返回值可序列化，天然适合进程隔离。
  - session 状态变更在锁内、解析 / 转储在锁外（`sessions.py`），不阻碍进程化。
- Why not primary: 100+ MiB 响应体每次跨进程 pickle 往返的 IPC 成本很可能抵消多核收益；PEP 734 子解释器需 Python 3.13+ 而 `setup.py` 要求 `python_requires=">=3.10"`；净收益与置信度均不及解析引擎替换。

## Synthesis Notes

若用户改选其它方向，有几条可与 Primary 叠加或前后接力：Alt-2「惰性不解析」本质就是 Primary 的核心子机制（`msgspec.Raw` 透传 `routed_experts` / `indexer_topk`），二者是同一条解析路径，无需当作互斥选项；Alt-4「并发准入」与 Primary 正交且互补，可在解析提速后作为防护层进一步压 p95 与峰值内存。Alt-1「增量协议」与 Alt-3「旁路通道」从源头削减 / 转移 payload，收益上限最高，适合作为协议侧的第二阶段——一旦 SGLang 契约可改，delta 或 ObjectRef 能把"根本不传 / 不解析大 blob"做到极致，届时本地解析提速退化为兜底层。Alt-5 则在 Python 升到 3.13+、或先 profiling 证明 IPC 收益为正后，可作为解析的并行执行后端替换 `ThreadPoolExecutor`。建议落地顺序：先 Primary（含 Alt-2 的 `Raw` 透传）拿到确定的本地收益，叠 Alt-4 控尾延迟，再评估 Alt-1 / Alt-3 的协议侧改造。
