# MiniMax lens — cross-system integration & data-flow tracing

**Stance:** integration-seam. Follow one user turn (audio ingress → VAD/endpointing → STT → LLM/tools → TTS →
egress) across every subsystem boundary and report where the wire is broken, missing, or silently lossy.

**Targets (read-only, core runtime only; no examples/docs/tests/vendored deps were read):**

| Framework | Path | Notes |
|---|---|---|
| LiveKit Agents | `path/to/livekit/agents/livekit-agents/livekit/agents` | Python core + `ipc/` process boundary |
| Pipecat | `path/to/voicekit/reference/pipecat/src/pipecat` | frame pipeline + `bus/` distributed bus |
| TEN Framework | `path/to/voicekit/reference/ten-framework` | Python runtime binding + `ai_agents/.../extension/` |
| Vocode | `path/to/voicekit/reference/vocode-core/vocode/streaming` | **stale** — HEAD `2024-11-15`, design reference only |

**Caveat carried into every Vocode finding:** the tree is ~2 years old and predates the current shape of all three
other frameworks. Vocode findings describe *design choices to avoid*, not current behaviour to copy.

**Method:** grep-driven seam tracing. For each candidate I confirmed the emit site, the (missing) consume site and
the failure path by reading the surrounding code. No claim below rests on a single grep hit alone.

---

## Findings (20)

### 1. The cross-process inference seam is unary — there is no streaming message type and no cancel message type
- **Severity:** blocks-NOW
- **Tag:** [api]
- **Seam:** LiveKit job subprocess → worker inference process (`ipc` unix-socket channel)
- **File:line (emit):** `ipc/proto.py:170-188` — `InferenceRequest{method, request_id, data: bytes}`;
  `ipc/proto.py:190-212` — `InferenceResponse{request_id, data: bytes | None, error}`.
  **Missing consume site:** `ipc/proto.py:264-279` — `IPC_MESSAGES` is the *complete* message catalogue and
  contains no chunk/stream/cancel variant. Server side: `inference_runner.py:35`
  `def run(self, data: bytes) -> bytes | None` (synchronous, whole-payload, whole-result), dispatched at
  `ipc/inference_proc_lazy_main.py:104-106` via `loop.run_in_executor(...)`.
- **Evidence:** `ipc/proto.py:264-279` lists ids 1..13 (`InitializeRequest` … `JoinTiming`) — none streaming;
  `ipc/proto.py:201` `channel.write_bool(b, self.data is not None)` — the payload is one opaque blob or nothing.
- **Fix sketch:** add `StreamChunk{request_id, seq, payload}` and `CancelRequest{request_id}` message types and
  make `run` an async-generator contract; a provider that streams partials must be able to cross this seam incrementally.
- **Why it matters for voicekit:** the kickoff constraint *"the boundary must not serialize a whole turn or a
  whole utterance"* is violated here by construction — this is the single most important seam in the comparison.

### 2. A recoverable provider error is dropped on the floor — no event, no metric, no log
- **Severity:** silent-data-loss
- **Tag:** [reliability]
- **Seam:** STT/LLM/TTS provider → session error policy → user & observers
- **File:line (consume site):** `voice/agent_session.py:2075-2079`
  `def _on_error(...)` → `if self._closing_task or error.recoverable: return` — returns **before** the counters
  at `:2082/2086/2090` and before any `error` event is emitted. The only emitters of an `"error"` event are
  `voice/agent_activity.py:2303, 2306, 2309, 2312`.
  **Missing consume site:** `voice/report.py:114-123` (`_error_counts`) counts *emitted* error events, so
  recoverable provider failures are absent from the session report.
- **Evidence:** `grep -n "_stt_error_counts" voice/agent_session.py` → `686` (init), `1390` (close), `2082` (inc),
  `2379` (reset). An error with `recoverable=True` touches none of them and emits no event.
- **Fix sketch:** emit a distinct `provider_error_recovered` event carrying provider + error type, and count it
  against a separate "degraded provider" budget that is visible in the report.

### 3. Any non-empty transcript resets the unrecoverable-error budget, so a flapping STT never trips it
- **Severity:** silent-data-loss
- **Tag:** [reliability]
- **Seam:** STT event → session error counter → session close policy
- **File:line:** `voice/agent_session.py:2376-2379`
  (`if ev.transcript: self._stt_error_counts = 0`) against the budget check at
  `voice/agent_session.py:2081-2084`
  (`if self._stt_error_counts <= self.conn_options.max_unrecoverable_errors: return`,
  default `max_unrecoverable_errors: int = 3` at `voice/agent_session.py:195`).
  The reset fires for **every** STT event including non-final interims — `_user_input_transcribed` is called
  from `voice/agent_activity.py:2366-2376` for interim and final alike.
- **Evidence:** STT error → partial transcript → STT error → partial transcript resets the counter to 0 each
  cycle. Compounded by finding #2 (the errors themselves are recoverable and never counted) and by
  `voice/audio_recognition.py:266-279`, where the STT pump retries `APIError` forever every 0.5 s
  (`_STT_RECONNECT_INTERVAL = 0.5`, `:60`) on a path that never touches the session budget: the session never
  closes and nothing is reported.
- **Fix sketch:** reset the counter only on a *final* transcript, or decay it (half-life) instead of zeroing it.

### 4. The avatar `clear_buffer` RPC fails open — the barge-in is silently dropped
- **Severity:** degraded-now
- **Tag:** [reliability]
- **Seam:** agent → remote avatar participant (LiveKit RPC channel)
- **File:line:** `voice/avatar/_datastream_io.py:229-239` — `_clear_buffer_task` issues
  `self._room.local_participant.perform_rpc(..., method=RPC_CLEAR_BUFFER, ...)` at `:232-236`; the handler is
  `except Exception as e:` at `:237`, `logger.error(...)` at `:238`, and `timeout = 0` at `:239` —
  *"mark playout done immediately if clear buffer rpc fails"*.
- **Evidence:** the failure path is **fail-open by construction** — the comment states the intent. The avatar
  keeps playing the pre-interruption audio over the user's turn, and the session still reports the interruption
  as delivered. The only trace is the log line inside the exception handler.
- **Fix sketch:** treat a failed clear-buffer as a failed interrupt: retry once, then emit `interrupted=False`
  on the session and mark the turn degraded instead of assuming success.

### 5. The LLM fallback adapter either truncates the turn or speaks the prefix twice
- **Severity:** silent-data-loss
- **Tag:** [bug]
- **Seam:** failing LLM provider → fallback LLM provider → TTS
- **File:line:** `llm/fallback_adapter.py:218-296` (`_try_generate` yields chunks as they arrive; a mid-stream
  failure re-raises and the caller switches providers at `llm/fallback_adapter.py:385-390`) and
  `llm/llm.py:354-355` (`if not self._retry_on_chunk_sent and self._event_ch.output_sent: e.retryable = False`).
  Adapter default `retry_on_chunk_sent: bool = False` at `llm/fallback_adapter.py:53`.
- **Evidence:** with the default, any failure *after the first chunk* logs
  *"failed after sending chunk, skip retrying"* (`llm/fallback_adapter.py:387-390`) → the user hears a truncated
  sentence and the turn dies. With `retry_on_chunk_sent=True` the retry re-emits from the beginning → the
  already-synthesized prefix is spoken twice, because nothing rolls back the chunks already yielded to the TTS.
- **Fix sketch:** on fallback, drop the partial assistant turn from the chat context and re-synthesize, or make
  the retry resume at a token offset that the TTS can splice.

### 6. The inference process handles requests one at a time — head-of-line blocking on the whole worker
- **Severity:** degraded-now
- **Tag:** [perf]
- **Seam:** inference process IPC reader loop → inference runner
- **File:line:** `ipc/inference_proc_lazy_main.py:85-95`
  (`async for msg in cch:` → `await self._handle_inference_request(msg)` **inline**, so the loop cannot service
  the next request, nor `ShutdownRequest`, nor `DumpStackTraceRequest` while one inference is in flight), and
  `ipc/inference_proc_lazy_main.py:104-106` (`await loop.run_in_executor(...)`).
  Contrast: the job-side executors *do* fan out — `ipc/job_proc_executor.py:128-131` and
  `ipc/job_thread_executor.py:341-344` both `asyncio.create_task(self._do_inference_task(msg))`.
- **Evidence:** the serialisation is asymmetric between the two sides of the same channel: the job process
  dispatches N concurrent inferences, the inference process executes them one after another.
- **Fix sketch:** `asyncio.create_task` per request in `entrypoint`, bounded by a semaphore, with a per-request
  deadline so one slow provider cannot starve the rest.

### 7. A missing required field in a bus payload becomes `None` instead of an error
- **Severity:** silent-data-loss
- **Tag:** [bug]
- **Seam:** Pipecat bus publisher process → bus subscriber process (Redis / pgmq)
- **File:line:** `bus/serializers/json.py:166-186` — for a dataclass target, every `init` field absent from the
  payload is filled with `init_kwargs[name] = None` at `:182` rather than raising.
- **Evidence:** `bus/messages.py:410-422` (`BusJobCancelMessage{job_id: str, reason: str | None}`) and
  `bus/messages.py:428-463` (the stream messages) declare `job_id: str` as required; a payload missing `job_id`
  deserializes into an object whose `job_id` is `None`, defeating the declared type — and the cancellation
  then silently targets nothing.
- **Fix sketch:** raise on a missing required field; only fields with a default may be absent. Cross-process
  schema validation belongs at the boundary, not at the type checker.

### 8. An unresolvable type nulls the field, and the whole message is then dropped without a trace
- **Severity:** silent-data-loss
- **Tag:** [reliability]
- **Seam:** JSON bus payload → typed message → subscriber
- **File:line:** `bus/serializers/json.py:151-158` (`_deserialize_typed`: `cls = _resolve_type(type_name)` at
  `:155`; `if cls is None:` at `:156` → `logger.warning(...)` at `:157` → `return None` at `:158`) and
  `bus/serializers/json.py:187-188` (`logger.warning("...no adapter registered for type {type_name}")` →
  `return None`). The drop happens at `bus/network/redis.py:104-109`
  (`except Exception:` at `:108` → `logger.exception(f"{self}: failed to deserialize message")` at `:109`).
- **Evidence:** for `BusFrameMessage` (`bus/messages.py:95-108`) the `frame` field becomes `None`; the message
  is never delivered to any subscriber, no error event is raised, no counter is incremented. A renamed or moved
  frame class (publisher on a newer version than the subscriber) therefore kills frames silently.
- **Fix sketch:** treat an unresolvable `__type__` as a hard error carrying the type name, and publish a
  `bus_message_dropped` event with a reason so the loss is countable.

### 9. The bus deserializer imports an arbitrary module path taken straight from the payload
- **Severity:** blocks-NOW
- **Tag:** [security]
- **Seam:** untrusted bus payload → Python import machinery
- **File:line:** `bus/serializers/json.py:198-215` — `_resolve_type` does
  `module = importlib.import_module(module_path)` at `:212`, where `module_path` is the `__type__` string taken
  from the payload (`:208`). There is no allowlist; the only guard is `except ImportError` at `:214`.
- **Evidence:** the bus channel is shared infrastructure — `bus/network/redis.py:88-96` publishes to a fixed
  channel name (`"pipecat:bus"` by default), and `bus/network/redis.py:96` publishes whatever it is handed.
  Anyone able to publish to that channel chooses which module every subscriber process imports.
- **Fix sketch:** resolve `__type__` only against a registry of known bus/frame modules (`pipecat.*`); reject
  anything else with a typed error.

### 10. Cancellation is carried on a transport that can lose it
- **Severity:** degraded-now
- **Tag:** [reliability]
- **Seam:** worker → worker over Redis (distributed bus)
- **File:line:** `bus/network/redis.py:88-96` (`publish` uses Redis **pub/sub** — no persistence, no delivery
  guarantee, no acknowledgement). The cancel messages are `bus/messages.py:204-216` (`BusCancelMessage`),
  `bus/messages.py:218-234` (`BusCancelWorkerMessage`) and `bus/messages.py:410-422` (`BusJobCancelMessage`).
- **Evidence:** the local bus's priority semantics *do* survive the hop — `bus/bus.py:169-187` routes system
  messages inline and `bus/queue.py:19-46` re-prioritises them on arrival — but a message published while the
  target subscriber is disconnected (restarting, network blip) is discarded by Redis with no retry and no
  dead-letter. A cancel that is lost leaves the remote worker running and the bot talking.
- **Fix sketch:** carry control messages (cancel/end) on a durable channel (Redis stream or list with ack) or a
  dedicated control connection, and add a cancel timeout with an idempotent retry.

### 11. There is no backpressure, no drop policy and no lag counter anywhere in the frame path
- **Severity:** degraded-now
- **Tag:** [scale]
- **Seam:** transport → processor → processor → output transport
- **File:line:** `processors/frame_processor.py:140-191` (`FrameProcessorQueue(asyncio.PriorityQueue)` — unbounded;
  `put()` at `:160-179` never blocks, so a slow consumer does not slow the producer);
  `transports/base_input.py:321` (`self._audio_in_queue = asyncio.Queue()` — unbounded);
  `utils/frame_queue.py:16-46` (`FrameQueue(asyncio.Queue)` — unbounded, used by the output transport via
  `transports/base_output.py:59`).
- **Evidence:** no drop policy and no lag counter exists anywhere in the core path. The consequence is silent:
  a stalled STT or TTS makes memory grow at the audio rate and the user hears audio that is seconds-to-minutes
  stale, with no signal that the pipeline is behind.
- **Fix sketch:** bounded audio rings with drop-oldest plus an exported lag counter — exactly the voicekit
  constraint; Pipecat has no equivalent today.

### 12. Assistant text arriving before the start frame is dropped, and an end frame without a start reports an empty completion
- **Severity:** silent-data-loss
- **Tag:** [bug]
- **Seam:** LLM provider → `LLMFullResponseAggregator` → `LLMContext`
- **File:line:** `processors/aggregators/llm_response.py:83-86`
  (`_handle_llm_text`: `if not self._started: return` — a `TextFrame` before `LLMFullResponseStartFrame` is
  discarded) and `processors/aggregators/llm_response.py:78-81` (`_handle_llm_end` does **not** check
  `self._started`, so it fires `on_completion(self._aggregation, True)` with an empty string).
- **Evidence:** `_handle_llm_text` is the only writer of `self._aggregation` and it is gated on `_started`, which
  is set only by `_handle_llm_start` (`:75-76`). A provider that emits text without a start frame loses the whole
  assistant turn, and the context aggregator then records an empty assistant message.
- **Fix sketch:** buffer pre-start text and flush it on start, or make the aggregator start-implicit
  (treat the first text as the start).

### 13. An uncaught exception in any Python extension calls `os._exit(1)` and kills the whole runtime process
- **Severity:** blocks-NOW
- **Tag:** [reliability]
- **Seam:** Python extension → C++ graph runtime
- **File:line:** `core/src/ten_runtime/binding/python/interface/ten_runtime/async_extension.py:481-500`
  (`_exit_on_exception` → `os._exit(1)` at `:500`), reached from every `_wrapper_on_*` at
  `async_extension.py:423-479` (e.g. `_wrapper_on_audio_frame` at `:473-479`).
- **Evidence:** the wrappers catch `BaseException` (`async_extension.py:426, 432, 438, 444, 450, 456, 462,
  470, 478`), so even `KeyboardInterrupt`/`SystemExit` inside a plugin callback terminates the process. One bad
  provider in one extension destroys every other extension, every graph and every session hosted in that
  process — the exact opposite of "plugins must be able to fail without killing the session".
- **Fix sketch:** catch `Exception` (not `BaseException`), report the failure into the graph as a failed
  extension/command, and let the runtime mark that one extension dead while the rest keep running.

### 14. Audio frames are silently dropped when the Python event loop is closed — and the two branches disagree
- **Severity:** silent-data-loss
- **Tag:** [bug]
- **Seam:** C++ graph → Python extension
- **File:line:** `async_extension.py:394-419` — `_proxy_on_audio_frame`. The multi-thread branch guards with
  `if self._ten_loop and not self._ten_loop.is_closed():` at `:413` and has **no `else`**: a missing or closed
  loop drops the frame with no log, no metric, no error. The single-thread branch (`:401-410`) has **no
  `is_closed()` check at all**, so it raises `RuntimeError` into the C++ callback instead.
- **Evidence:** the same pattern is repeated for every callback — `:226, 242, 268, 294, 318, 339, 361, 386`
  (cmd, data, video, audio) — so the same silent drop exists for messages, not just audio.
- **Fix sketch:** log and count the drop, and make both branches symmetric (check `is_closed()` in both).

### 15. One un-tracked cross-thread coroutine hop per audio frame, with no backpressure
- **Severity:** degraded-now
- **Tag:** [perf]
- **Seam:** C++ graph → Python extension → Python event loop
- **File:line:** `async_extension.py:401-419` —
  `asyncio.run_coroutine_threadsafe(self._wrapper_on_audio_frame(self._async_ten_env, audio_frame), main_loop)`
  at `:407-410`. The returned `concurrent.futures.Future` is **discarded**: nothing can cancel a queued frame,
  observe completion, or bound the number in flight.
- **Evidence:** at 10 ms frames that is 100 un-tracked coroutines per second per extension; if the Python
  callback is slower than realtime the loop's task set grows without bound, because each
  `run_coroutine_threadsafe` creates a new task. There is no queue depth, no drop and no lag counter anywhere
  on this path. In single-thread mode every extension shares one loop
  (`async_extension.py:177-187`, `global_thread_manager.py:15-42`), so one slow extension stalls all of them.
- **Fix sketch:** retain the futures in a bounded set, drop the oldest on overflow, and export a lag counter —
  the same bounded-ring-with-lag-counter shape voicekit requires.

### 16. The wire-format validator is written entirely with `assert`, so `python -O` removes every bounds check
- **Severity:** blocks-NOW
- **Tag:** [security]
- **Seam:** C++ runtime ↔ Python (binary `Value` serialization)
- **File:line:** `core/src/ten_runtime/binding/python/interface/ten_runtime/value_buffer.py:277-298`
  (`_validate_buffer_header` checks magic, version, type id and length with `assert False, "..."` at lines 280,
  287, 290, 293, 296) and `value_buffer.py:301-424` (the same for every field: `:308, 312, 318, 324, 330, 338,
  352, 358, 373, 395, 400, 407, 420-422`). `serialize_to_buffer` asserts the final position matches the
  computed size at `value_buffer.py:269-272`.
- **Evidence:** under `python -O` / `PYTHONOPTIMIZE=1` every one of these assertions is removed, so a truncated or
  hostile buffer is *parsed* rather than rejected. This is the Python equivalent of the `unwrap()` the voicekit
  constraint forbids on network/audio paths.
- **Fix sketch:** raise explicit typed exceptions on every path that parses bytes from another process; reserve
  `assert` for internal invariants that are not reachable from untrusted input.

### 17. The turn id is maintained by the turn detector but never crosses the seam
- **Severity:** silent-data-loss
- **Tag:** [bug]
- **Seam:** TEN turn-detection extension → downstream LLM extension
- **File:line (write site):**
  `ai_agents/agents/ten_packages/extension/ten_turn_detection/extension.py:27` (init `self.next_turn_id = 1`)
  and `:133` (`self.next_turn_id += 1`) — incremented for every decided turn.
  **Missing read site:** the emitted payload at `extension.py:142-145` carries only `"text"` and `"is_final"`;
  `grep -rn "next_turn_id" --include='*.py'` across the whole TEN tree returns exactly those two lines —
  **nothing ever reads it**.
- **Evidence:** a write-path with no read-path. The downstream consumer cannot tell turn N from turn N+1, so a
  late-arriving final transcript from turn N is indistinguishable from turn N+1's, and no specific turn can be
  cancelled or superseded.
- **Fix sketch:** put the turn id in the emitted `Data` (e.g. `out_data.set_property_int("turn_id", ...)`) and
  require it on the consumer side.

### 18. The interrupt path cancels the agent but not the worker that actually writes audio
- **Severity:** degraded-now
- **Tag:** [bug]
- **Seam:** interrupt → synthesis-results worker → output device (Vocode)
- **File:line:** `vocode/streaming/streaming_conversation.py:824-847` (`broadcast_interrupt`) cancels
  `self.agent` (`:843`), `self.agent_responses_worker` (`:844`) and `self.actions_worker` (`:845-846`) — but
  **not** `self.synthesis_results_worker`, which exists (`:647`), is started (`:698`) and is terminated only at
  `:1035`.
- **Evidence:** the synthesis-results worker is the one that pushes `SynthesisResult`s to the output device
  (`vocode/streaming/streaming_conversation.py:538-580`), so a synthesis chunk computed before the interrupt can
  still reach the output device after the user has started speaking. `InterruptibleWorker._run_loop`
  (`vocode/streaming/utils/worker.py:203-232`) only skips items that were *already* marked interrupted.
- **Fix sketch:** cancel/drain the synthesis-results worker in `broadcast_interrupt`, and stamp every queued
  item with the generation it belongs to so stale generations can be discarded.

### 19. The backchannel buffer is written but only drained on a *later* final transcript — it is lost on termination
- **Severity:** silent-data-loss
- **Tag:** [reliability]
- **Seam:** STT → transcript (Vocode)
- **File:line (write site):** `vocode/streaming/streaming_conversation.py:245`
  (`self.human_backchannels_buffer.append(transcription)`).
  **Read site:** `vocode/streaming/streaming_conversation.py:291-297` — the only drain, and it runs only when a
  *non-ignored* transcription arrives. `vocode/streaming/streaming_conversation.py:1006-1013`
  (`terminate()`) awaits `broadcast_interrupt()` and then publishes
  `TranscriptCompleteEvent(conversation_id=self.id, transcript=self.transcript)` at `:1010-1013`
  **without** flushing the buffer.
- **Evidence:** a user backchannel ("mm-hmm") that is followed by the call ending is silently absent from the
  published transcript — the buffer is written and never read.
- **Fix sketch:** flush `human_backchannels_buffer` into the transcript inside `terminate()` before publishing
  `TranscriptCompleteEvent`.

### 20. One call has three parallel identities and two propagation channels, one of which leaks
- **Severity:** silent-data-loss
- **Tag:** [bug]
- **Seam:** telephony server → conversation → agent → Redis stream (Vocode)
- **File:line:** `vocode/streaming/models/telephony.py:44-52` (`CreateInboundCall` carries `vonage_uuid`,
  `twilio_sid` **and** `conversation_id`, all `Optional[str] = None`);
  `vocode/streaming/streaming_conversation.py:598-599`
  (`self.id = conversation_id or create_conversation_id()`, then `ctx_conversation_id.set(self.id)`);
  `vocode/streaming/telephony/conversation/abstract_phone_conversation.py:54` sets the same contextvar again.
  `vocode/__init__.py:18-33` — `ContextWrapper.set` keeps **one** token per wrapper (`self.__token`), so the
  second `set` overwrites the first's token and `reset()` can no longer undo it.
- **Evidence:** the same id travels two ways (explicit parameter at `streaming_conversation.py:294, 313` and via
  the contextvar). The contextvar's token clobbering means the id leaks into whatever runs next in that context
  and the Sentry tag is overwritten (`vocode/__init__.py:26-31`).
- **Fix sketch:** one id type and one propagation channel (explicit); per-call tokens, not one token per wrapper.

---

### Supplementary observations (verified, not counted as findings)

These were confirmed with the same rigour and are worth carrying into synthesis as lower-severity notes:

- LiveKit `ipc/inference_proc_lazy_main.py:99-101` — an unknown inference method logs a warning and then
  **falls through** to `self._runners[msg.method].run` at `:105` → `KeyError` → caught at `:108` → the client
  receives `error=str(e)` at `:110`, i.e. the literal `"'stt_v2'"`. The client turns that into
  `RuntimeError(f"inference of {method} failed: {inf_resp.error}")` (`ipc/job_proc_lazy_main.py:176-177`).
- LiveKit `voice/agent_session.py:771` (`self._recorded_events: list[AgentEvent] = []`),
  `:795-796` (`emit` appends every event), reset only at session start `:1013`; consumed at `job.py:481`
  (`events=session._recorded_events`). The list is append-only for the life of the session and
  `make_session_report` serializes all of it.
- LiveKit `voice/events.py:491` — `SpeechCreatedEvent.speech_handle` is `Field(..., exclude=True)`, so
  `voice/report.py:72` (`event.model_dump()`) drops the speech id from every serialized `speech_created`
  event: a client cannot correlate which speech was created.
- LiveKit clock discipline — `voice/agent_activity.py` uses `time.time()` 30× and `time.perf_counter()` 3×;
  `voice/audio_recognition.py` uses `time.time()` 20× and **zero** `time.monotonic()`;
  `stt/fallback_adapter.py:273, 303, 364, 476` compute durations as `time.time() - start_time`. The inference
  process times initialization with `time.perf_counter()` (`ipc/inference_proc_lazy_main.py:70, 78`). A wall-clock
  step (NTP) makes a measured latency negative, and a duration produced in the job process cannot be compared
  with one produced in the inference process.
- LiveKit `voice/agent_session.py:2213-2231` — `_record_loop_stall` records count/total/max of blocked event-loop
  time on the session span. Strong capability; noted here only because the counters are per-session in-memory
  and are not exported as a metric.
- Pipecat `bus/serializers/json.py:104-113` — the `BaseModel` branch serializes `value.__dict__` only, so
  pydantic `extra` fields and private attributes are silently dropped across the bus.
- Pipecat `bus/serializers/json.py:114-115` — any callable attribute is serialized as `null`.
- Pipecat `bus/adapters/llm_context_adapter.py:71`
  (`messages = [self._deserialize_message(m, deserialize_value) for m in data["messages"]]` — a payload without
  `"messages"` raises `KeyError`), which propagates into `bus/network/redis.py:104-109` → the message is dropped.
  Note the asymmetry: the *optional* `tools`/`tool_choice` keys are guarded (`:73-77`) but the *required*
  `messages` key is not.
- Pipecat `bus/network/redis.py:104-109` — every deserialization failure in the distributed bus ends in
  `logger.exception` and a dropped message: no metric, no event, no dead-letter.
- Pipecat `services/llm_service.py:612` — `raise NotImplementedError(f"run_inference() not supported by
  {self.__class__.__name__}")` is the only guard on an optional capability implemented by a handful of the 60+
  providers; the failure surfaces inside a request path rather than at wiring time.
- Pipecat `workers/llm/backend_llm_worker.py:570, 585, 603` — cross-worker request correlation uses a
  **positional index** (`_request_awaiting_run`) into a mutable message list rather than an id; if the context is
  truncated or summarized in between, the pending request either never fires or fires against the wrong turn.
- TEN `global_thread_manager.py:15-42` — `GlobalThreadManager` is a process-wide singleton with a single event
  loop shared by every extension and every session: **zero per-session isolation**.
  `async_extension.py:177-187` hands that same loop to every extension.
- TEN `global_thread_manager.py:94` — `self._loop_ready_event.wait()` has **no timeout**; if `_thread_routine`
  raises before setting it (`:116` catches only `ImportError`), graph configuration blocks forever.
- TEN `addon_manager.py:30-63` — an addon registration failure is `print`ed and ignored (`:55-61`), never
  reaching the TEN logger and never failing the graph load; `addon_manager.py:63` clears `_registry` after the
  first pass, and `_registered_addons` (`:39, 54`) makes a **second graph in the same process register nothing at
  all** — a silently extension-less second graph. `addon_manager.py:87-88` and `:130` repeat the
  print-and-continue pattern.
- TEN `ten_turn_detection/extension.py:77-81` — `try: is_final, _ = data.get_property_bool("is_final") except
  Exception as _e: pass` → on failure `is_final` stays `False` → `extension.py:100-101` returns → the final
  transcript never reaches `_eval_decision` (`:106`) and the LLM never receives a final turn. The user speaks, the
  bot stays silent, and the exception was discarded by `except ... : pass`.
- TEN `ten_turn_detection/extension.py:127-138` — `_process_new_turn` clears `self.cached_text` *before* the
  `Wait` branch returns without sending, so a "wait" decision drops the accumulated transcript; the next
  utterance's payload (`extension.py:88`) then contains only the new text.
- TEN `streamid_adapter/extension.py:43` — `stream_id, _ = frame.get_property_int("stream_id")` discards the
  `TenError` returned by `ten_env.py:111` (`get_property_int(self, path: str) -> tuple[int, TenError | None]`),
  then `extension.py:45-52` writes `{"session_id": f"{stream_id}"}` into the frame metadata. A frame without the
  property yields the session id `"None"` (or `"0"`) for **every** frame, with no warning — two concurrent
  streams that lack the property are indistinguishable downstream.
- TEN `ten_vad_python/extension.py:24-25` (`BYTES_PER_SAMPLE = 2`, `SAMPLE_RATE = 16000`),
  `:139-145` (`_send_audio_frame` sets bytes-per-sample / sample-rate / channels unconditionally),
  `:165` (`audio_frame.get_buf()` — the incoming frame's own rate and channel count are never read),
  `:173-179` (hop math in samples), `:183` (`np.frombuffer(audio_buf, dtype=np.int16)`). At a 24 kHz or 8 kHz
  transport the outgoing header lies, the hop math is wrong, and the ASR downstream resamples against the wrong
  rate — a silently wrong transcription, not a crash.
- Vocode `vocode/streaming/agent/base_agent.py:516-533` — the transfer action's call id is guarded by
  `assert ... is not None` (lines 517-519, 527-529); under `python -O` a `None` id flows straight into
  `vocode/streaming/telephony/client/twilio_client.py:82-84`, which builds a REST URL with it.
- Vocode `vocode/streaming/utils/worker.py:37-46, 57-73, 78-95` and
  `vocode/streaming/utils/events_manager.py:13, 20-21` — every queue between transport and TTS is unbounded
  (`asyncio.Queue()` / `janus.Queue()`), `consume_nonblocking` uses `put_nowait`, and `publish_event` silently
  discards any event type not in `subscriptions`.
- Vocode `vocode/streaming/telephony/server/base.py:132-134, 150-151` vs
  `vocode/streaming/telephony/server/router/calls.py:113` — outbound calls mint the conversation id server-side
  but the inbound path accepts a caller-supplied one, which then becomes the Redis stream key at
  `vocode/streaming/utils/redis_conversation_message_queue.py:21, 27-28`.

---

## Dimension × framework matrix

Winner per dimension, with the reason and at least one `file:line` anchor per framework. "No clear winner" is
used where the evidence does not separate the options.

### 1. Transport & audio path — **Winner: no clear winner**
LiveKit has the most explicit interruption plumbing (`voice/room_io/_output.py:171`,
`voice/avatar/_datastream_io.py:171`); Pipecat has the cleanest framing contract
(`processors/frame_processor.py:140-191`, `transports/base_output.py:147-151`); TEN has the most explicit
per-frame wire type (`core/src/ten_runtime/binding/python/interface/ten_runtime/value_buffer.py:239-274`) but
mislabels the sample rate (`ten_vad_python/extension.py:139-145`); Vocode has the weakest path
(`vocode/streaming/utils/worker.py:37-46` — unbounded everywhere). LiveKit's fail-open avatar RPC
(`voice/avatar/_datastream_io.py:237-239`) cancels Pipecat's advantage; TEN's hard-coded rate header
(`ten_vad_python/extension.py:24-25`) cancels its advantage. **No clear winner.**

### 2. Concurrency & isolation — **Winner: LiveKit Agents**
LiveKit isolates a job into its own OS process (`worker.py:133-137` selects `PROCESS` or `THREAD`;
`ipc/job_proc_executor.py:99-122` creates it) and isolates inference into a separate process
(`worker.py:615-620`), so a blocking provider cannot stall other sessions. Pipecat is a single event loop with
per-processor tasks (`processors/frame_processor.py:1270-1277`) and no per-session isolation by default.
TEN is the weakest: one process-wide event loop for every extension and session
(`global_thread_manager.py:15-42`, `async_extension.py:171-187`). Vocode is a single loop with per-worker tasks
(`vocode/streaming/utils/worker.py:203-232`).

### 3. Turn-taking — **Winner: LiveKit Agents**
LiveKit has an explicit endpointing model (`voice/endpointing.py:10-14`), a false-interruption recovery event
(`voice/agent_activity.py:5286-5287`), and a dedicated turn-commit path
(`voice/audio_recognition.py:1119-1120`). Pipecat has an explicit interruptible-frame contract
(`processors/frame_processor.py:1161-1188`, `turns/user_turn_strategies.py`) but no false-interruption recovery.
TEN's detector silently loses the turn id (`ten_turn_detection/extension.py:133`) and can stall on a failed
`is_final` read (`ten_turn_detection/extension.py:77-81`). Vocode's interrupt path does not cancel the
audio-writing worker (`vocode/streaming/streaming_conversation.py:824-847`).

### 4. Provider abstraction & plugin contract — **Winner: no clear winner**
LiveKit's contract is the smallest but also the least capable: `inference_runner.py:16-37` (a class-level
registry + one synchronous method), with fallback composition as an adapter
(`llm/fallback_adapter.py:38-82`, `stt/fallback_adapter.py:360-420`). Pipecat has the richest contract — 60+
services over `services/ai_service.py:33-40` and `services/llm_service.py:307-612` — but a provider author must
also pick a frame vocabulary and an adapter (`adapters/llm/base_llm_adapter.py`), and optional capabilities fail
at runtime (`services/llm_service.py:612`). TEN's contract is the cleanest to *declare*
(`addon_manager.py:100-143` + a `manifest.json`) but the least forgiving at *runtime*
(`async_extension.py:481-500`). Vocode's is the loosest
(`vocode/streaming/synthesizer/base_synthesizer.py`). **No clear winner** — but TEN's manifest-plus-one-
registration is the shape voicekit should borrow.

### 5. State & context management — **Winner: no clear winner**
LiveKit has the richest model (typed chat items, ids, fingerprints, provider-format projection) at
`llm/chat_context.py:529-1080` plus a remote variant at `llm/remote_chat_context.py:17-103`. Pipecat has the
simplest (`processors/aggregators/llm_context.py:113-300`) and the only one with a documented cross-process
serializer (`bus/adapters/llm_context_adapter.py:32-78`) — which is also its weakest link (#7, #12).
TEN has no conversation-state concept at all: state is whatever the graph carries in `Data` properties
(`ten_turn_detection/extension.py:119-122`). Vocode has three ids for one call
(`vocode/streaming/models/telephony.py:44-52`). **No clear winner.**

### 6. Failure handling — **Winner: Pipecat**
Pipecat pushes errors *up the pipeline* as first-class frames with a category and a `fatal` flag
(`processors/frame_processor.py:893-900`), so application code can react and the pipeline keeps running
(`services/ai_service.py:284-289`). LiveKit drops recoverable errors entirely
(`voice/agent_session.py:2075-2079`) and lets a flapping provider reset its own error budget
(`voice/agent_session.py:2376-2379`). TEN kills the process (`async_extension.py:481-500`). Vocode uses `assert`
for input validation (`vocode/streaming/agent/base_agent.py:516-533`). Pipecat's own weakness is that the
*bus* swallows errors (`bus/network/redis.py:104-109`), but the in-pipeline contract is clearly the best of the four.

### 7. Observability — **Winner: LiveKit Agents**
LiveKit has per-turn latency attribution (`metrics/base.py:20-35` — `duration`, `ttft`, `cancelled`, token
counts) *and* explicit event-loop blocking detection (`voice/agent_session.py:2213-2231` — count, total and max
stall). Pipecat has a heartbeat frame that traverses the whole pipeline and measures round-trip time
(`pipeline/worker.py:1734-1745`) plus TTFB metrics (`processors/frame_processor.py:518-535`) — good liveness, but
no stall *attribution* to a processor. TEN logs to the TEN logger only (`async_extension.py:484-488`).
Vocode relies on Sentry spans (`vocode/streaming/streaming_conversation.py:854-872`) and an event manager that
silently drops unsubscribed types (`vocode/streaming/utils/events_manager.py:20-21`). LiveKit's mixed clock
discipline is the one mark against it.

### 8. Extensibility cost — **Winner: TEN Framework**
TEN: implement one extension class, decorate it with `register_addon_as_extension`
(`addon_manager.py:100-143`), ship a `manifest.json` — **no core edits**
(`packages/core_extensions/default_async_extension_python/addon.py`). Pipecat: implement a service class plus
pick frames and an adapter (`services/ai_service.py:33-40`, `bus/serializers/json.py:55-62`). LiveKit: implement
a provider *and* register an inference runner (`inference_runner.py:19-27` — and the runner registry is captured
**once** at worker start, `worker.py:615-620`, so a late registration is invisible to the inference process).
Vocode: subclass the synthesizer/transcriber base and accept the factory wiring
(`vocode/streaming/synthesizer/abstract_factory.py`). TEN wins on "no core edits", loses on runtime robustness (#13).

### 9. Deployment & scale — **Winner: no clear winner**
LiveKit separates the job process from the inference process (`worker.py:615-620`,
`ipc/job_proc_executor.py:99-122`) and can run jobs in-process or out-of-process (`worker.py:133-137`), but the
inference process serializes requests (#6) and the session event buffer grows without bound
(`voice/agent_session.py:771, 795-796`). Pipecat's worker is a single loop per process (`pipeline/worker.py:211`)
and its queues are unbounded (#11), but it has a distributed bus with durable backends
(`bus/network/pgmq.py`, `bus/network/redis.py`). TEN is a graph runtime with native extensions
(`global_thread_manager.py:15-42`) but a single shared Python loop and an `os._exit` blast radius. Vocode is the
least scalable — unbounded queues (`vocode/streaming/utils/worker.py:37-46`) and a Redis stream keyed on a
caller-supplied id (`vocode/streaming/utils/redis_conversation_message_queue.py:21`). **No clear winner.**

### 10. Language boundary — **Winner: no clear winner**
LiveKit's boundary is a strict unary RPC with a hand-rolled length-prefixed framing
(`ipc/proto.py:170-212`, `ipc/channel.py:62-80`) — small, fast and **incapable of streaming or cancellation**
(#1). Pipecat's is a typed JSON bus with pluggable adapters (`bus/serializers/json.py:64-148`) that *can* carry
chunks (`bus/messages.py:428-463` defines start/data/end stream messages) but resolves types by importing a
module path from the payload (#8, #9) and turns missing required fields into `None` (#7). TEN's is the most
binary and the most versioned — a magic/version/type header (`value_buffer.py:239-274`) — but the validator is
`assert`-based (#16) and the callback failure mode is `os._exit` (#13). Vocode has no real boundary; it is a
Redis stream keyed by a string id (`vocode/streaming/utils/redis_conversation_message_queue.py:19-35`).
**No clear winner** — but TEN's magic/version/type header is the right shape, and Pipecat's start/data/end message
triple is the right streaming shape; voicekit needs both.

---

## Cross-cutting notes for synthesis

1. **The streaming-across-the-boundary requirement is the sharpest discriminator.** LiveKit's IPC cannot do it
   (#1); Pipecat's bus defines exactly the right message triple (`bus/messages.py:428-463`) but no framework
   wires a *provider* stream through it; TEN's binary protocol can (`value_buffer.py:239-274`) but the callback
   path drops frames (#14) and has no backpressure (#15). voicekit needs an explicit
   `StreamStart / StreamChunk{seq} / StreamEnd / Cancel` message set with a monotonic sequence number so a
   consumer can detect loss.

2. **Every framework that crosses a process boundary loses cancellation, and every one that keeps cancellation
   loses something else.** LiveKit: no cancel message (#1). Pipecat: cancel over lossy pub/sub (#10).
   TEN: no cancel concept, plus `os._exit` (#13). Vocode: cancellation reaches the agent but not the audio
   writer (#18). Cancellation must be a *first-class message* on the same transport as the data, with an
   idempotent retry and a deadline.

3. **Silent drops are the dominant failure mode across all four.** LiveKit drops recoverable errors (#2) and
   resets its error budget on recovery (#3); Pipecat nulls missing fields (#7) and drops unresolvable types
   (#8); TEN discards property-read errors (`ten_turn_detection/extension.py:77-81`) and drops frames on a
   closed loop (#14); Vocode drops backchannels on termination (#19) and uses `assert` for validation
   (`vocode/streaming/agent/base_agent.py:516-533`). **The voicekit rule should be: no failure may cross a
   boundary without being either (a) represented in the wire message, or (b) counted in a metric.**

4. **No framework has a bounded audio ring with a lag counter.** LiveKit's queueing is inside the LiveKit SDK
   (`voice/io.py:511`), Pipecat's are unbounded (`processors/frame_processor.py:140-191`,
   `transports/base_input.py:321`), TEN's per-frame hop is untracked (`async_extension.py:407-410`), Vocode's are
   unbounded (`vocode/streaming/utils/worker.py:37-46`). This is an unqualified gap, and the voicekit constraint
   (bounded rings, drop-oldest, lag counter) has no precedent to copy — it must be designed.

5. **Clock discipline is inconsistent everywhere and is directly measurable in the traces.** LiveKit mixes
   `time.time()` with `time.perf_counter()` across the IPC seam; Pipecat is the only framework with an explicit
   clock abstraction (`clocks/system_clock.py:30-38`, monotonic nanoseconds) — worth adopting verbatim; TEN and
   Vocode use wall clock throughout. The voicekit constraint ("monotonic clock for all durations") should be
   non-negotiable at the boundary.

6. **Provider plugins cannot fail independently in any of the four.** TEN is the extreme case (`os._exit`,
   #13); LiveKit lets a plugin's failure close the whole session (`voice/agent_session.py:2075-2096`); Pipecat
   is the only one that separates a non-fatal error from a fatal one at the frame level
   (`processors/frame_processor.py:893-900`). voicekit's "plugin fails without killing the session" requirement
   should adopt Pipecat's error taxonomy and TEN's `os._exit`-free manifest registration.

7. **Id correlation across the seam is broken everywhere, in three distinct ways:** a counter that is written
   but never emitted (TEN turn id, #17); an optional id read with its error discarded, producing a literal
   `"None"` (`streamid_adapter/extension.py:43`); and several parallel ids for one entity (Vocode's
   `vonage_uuid`/`twilio_sid`/`conversation_id`, #20). The voicekit rule should be: **exactly one id type per
   entity, carried in every message, and every id read must have its error checked.**
