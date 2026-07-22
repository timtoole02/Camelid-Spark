# Live Inference Telemetry

Camelid's server exposes a real-time event stream describing inference as it
executes. The web UI's **Inference Observatory** tab renders this stream; any
SSE client can consume it.

```
GET /api/telemetry/stream        (Server-Sent Events)
```

Each SSE message is named `telemetry` and carries one JSON object. The first
message is a hello (`{"event":"hello","schema":"camelid.telemetry/v1",...}`);
every subsequent message is an envelope:

```json
{
  "seq": 17,
  "t_ms": 237960,
  "request_id": "49cb98e3-…",
  "model_id": "Llama 3.2 3B Instruct",
  "event": "decode_started",
  "context_position": 15
}
```

`seq` orders events, `t_ms` is milliseconds since the server started emitting,
and `request_id`/`model_id` attribute the event to a generation request.

## Truthfulness contract

Every event is emitted from a real code path doing real work (see
`src/telemetry.rs`). There is no synthetic, replayed, or decorative event
source anywhere in the stack:

- An idle server sends nothing besides the hello and SSE keep-alive comments.
- Consumers must not animate inference activity without a backing event; the
  Observatory's renderer modules only react inside their event handlers
  (`frontend/scripts/observatory-smoke.mjs` enforces this).
- If the stream drops events under load, a `lagged` notice is emitted rather
  than papering over the gap.

## Event vocabulary (`camelid.telemetry/v1`)

| Event | Emitted when | Notable fields |
| --- | --- | --- |
| `inference_started` | A generation request enters the engine | `backend`, `quantization`, `architecture`, `prompt_tokens`, `max_tokens`, `context_length`, `temperature`, `stream` |
| `inference_finished` | The request closes (any outcome) | `status` (`ok`/`error`/`disconnected`), `finish_reason`, `completion_tokens`, `total_ms`, `ttft_ms`, `decode_tps` |
| `inference_error` | A real failure occurs while a generation is active | `code`, `message` |
| `prefill_started` | Prompt evaluation begins | `prefill_tokens`, `path`, `layers_total` |
| `prefill_progress` | A prefill chunk really completed | `tokens_done`, `tokens_total` |
| `decode_started` | The first generated token's forward pass begins | `context_position` |
| `layer_started` / `layer_completed` | A transformer layer executes on a CPU lane | `layer`, `layers_total`, `duration_us` |
| `token_decoded` | The sampler (CPU or GPU lane) produced a token | `token_id`, `context_position`, `layers_total` |
| `kv_cache_updated` | The KV cache advanced | `position`, `capacity`, `approx_bytes` |
| `sampler_step` | A decode step sampled from real logits | `chosen_token_id`, `mode`, `candidates` (top-8 post-softmax) |
| `receipt_written` | A parity receipt was sealed server-side | `receipt_id`, `reproducible`, `gguf_sha256` |
| `worker_node_active` / `worker_node_idle` / `worker_node_error` | A distributed-worker TCP roundtrip starts / lands / fails | `node`, `detail`, `error` |

## Lane coverage and granularity

- **Llama serve lane** (primary): full vocabulary above. On GPU-resident
  prefill/decode paths the engine has no per-layer visibility (layers run
  inside one Metal command buffer), so `layer_*` events are simply absent
  there — consumers should pace layer visuals from real `token_decoded`
  cadence and `layers_total` instead. CPU lanes emit real `layer_*` events.
- **Gemma 4 serve lane**: lifecycle events plus one `token_decoded` pulse per
  really-generated token. Prompt token counts and context length are not
  reported by that runtime and are recorded as `0` ("not reported"), never
  estimated.
- **Speculative decode** (`CAMELID_SPEC_DECODE`, default off): tokens accepted
  in a speculation batch do not emit individual `token_decoded` events.

## Rate limiting and cost

High-frequency classes are throttled server-side (layer events ≥15 ms apart,
KV ≥50 ms, sampler ≥80 ms, prefill progress ≥33 ms, worker active/idle
≥100 ms); lifecycle, token, receipt, and error events always pass. With no
subscriber connected, every emit short-circuits on one atomic load, so
serving performance is unaffected when nothing is watching. Sampler top-k
softmax is computed only while a subscriber is connected.
