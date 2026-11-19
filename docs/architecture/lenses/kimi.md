# Kimi Code Refactors — voicekit

**Lens stance:** long-context code-level refactor specialist. Read the load-bearing files of
each framework and propose *specific rewrites* — concrete diffs, not abstract advice.

**Scope inspected (core runtime only; no examples, docs, tests, or vendored deps were used as
evidence for a claim):**

| Framework | Path | Core surface read |
|---|---|---|
| LiveKit Agents | `path/to/livekit-agents/livekit-agents/livekit/agents` | `stt/stt.py`, `stt/fallback_adapter.py`, `stt/stream_adapter.py`, `llm/llm.py`, `llm/fallback_adapter.py`, `llm/chat_context.py`, `tts/tts.py`, `vad.py`, `voice/agent_activity.py`, `voice/agent_session.py`, `voice/generation.py`, `voice/endpointing.py`, `voice/io.py`, `voice/room_io/_input.py`, `voice/room_io/_output.py`, `utils/aio/channel.py`, `utils/aio/itertools.py`, `utils/aio/duplex_unix.py`, `ipc/proto.py`, `ipc/channel.py`, `ipc/proc_pool.py`, `worker.py`, `job.py`, `metrics/base.py`, `plugin.py`; one reference plugin (`livekit-plugins/livekit-plugins-deepgram/.../stt.py`) for provider-author cost |
| Pipecat | `path/to/pipecat/src/pipecat` | `processors/frame_processor.py`, `utils/frame_queue.py`, `pipeline/pipeline.py`, `pipeline/worker.py`, `pipeline/parallel_pipeline.py`, `transports/base_input.py`, `transports/base_output.py`, `services/ai_service.py`, `services/stt_service.py`, `services/tts_service.py`, `services/websocket_service.py`, `services/cartesia/tts.py`, `services/deepgram/stt.py`, `turns/user_turn_controller.py`, `turns/user_turn_strategies.py`, `turns/user_turn_processor.py`, `observers/base_observer.py`, `processors/metrics/frame_processor_metrics.py` |
| TEN Framework | `path/to/ten-framework/core/src/ten_runtime` | `binding/python/interface/ten_runtime/{async_extension.py,ten_env.py,async_ten_env.py,audio_frame.py,global_thread_manager.py}`, `binding/python/native/extension/extension.c`, `binding/python/native/msg/audio_frame.c`; `packages/core_extensions/default_llm_extension_python/extension.py`; `ai_agents/agents/ten_packages/extension/assemblyai_asr_python/{extension.py,audio_buffer_manager.py}` |
| Vocode | `path/to/vocode-core/vocode` | `streaming/streaming_conversation.py`, `streaming/transcriber/base_transcriber.py`, `streaming/transcriber/deepgram_transcriber.py`, `streaming/synthesizer/base_synthesizer.py`, `streaming/output_device/*`, `streaming/input_device/microphone_input.py`, `streaming/utils/worker.py`, `streaming/agent/base_agent.py`, `streaming/agent/openai_utils.py`, `streaming/agent/token_utils.py`, `streaming/models/transcript.py`, `utils/sentry_utils.py` |

**Caveat carried throughout:** Vocode is a stale tree (last release ~2023). Claims about it are
design-reference claims about the shape of the code, and specific findings about it are marked
as *stale* where the code is provably dead.

---

## Kimi Code Refactors — voicekit

### refactor-1: Bounded audio ring with drop-oldest and a lag counter

**File**: `livekit-agents/livekit/agents/vad.py:105` (same pattern at `stt/stt.py:456`,
`tts/tts.py:562`, `llm/llm.py:254`; the unbounded primitive is
`utils/aio/channel.py:50-53` + `:154-158`)

**Why it scales**: every audio frame entering the runtime is appended to an unbounded deque.
When a provider stalls, memory grows linearly with stall time and the audio that eventually
arrives is stale by the whole stall — unbounded lag instead of bounded lag.

**Before** (`livekit-agents/livekit/agents/utils/aio/channel.py:50-53` and `:154-158`):

```python
class Chan(Generic[T]):
    def __init__(self, maxsize: int = 0, loop=None) -> None:
        self._maxsize = max(maxsize, 0)          # 0 == unbounded, by default

    def full(self) -> bool:
        if self._maxsize <= 0:
            return False                         # never full -> never blocks
        else:
            return self.qsize() >= self._maxsize
```

```python
# vad.py:105 — the VAD hot path, unbounded
self._input_ch = aio.Chan[rtc.AudioFrame | VADStream._FlushSentinel]()
```

**After** (the Rust core's ring; the Python plugin side mirrors the same contract):

```rust
/// Bounded audio ring. Drop-oldest on overflow, with a monotonic lag counter
/// that the session reports as `audio.lag_frames`.
pub struct AudioRing<T> {
    buf: VecDeque<T>,
    cap: usize,
    dropped: u64,
}

impl<T> AudioRing<T> {
    pub fn new(cap: usize) -> Self {
        Self { buf: VecDeque::with_capacity(cap), cap, dropped: 0 }
    }

    /// Pushes one item. On overflow the OLDEST item is dropped and the lag
    /// counter advances; the caller can read `lag()` to report the loss.
    pub fn push(&mut self, v: T) -> Result<(), u64> {
        let mut dropped = 0;
        if self.buf.len() == self.cap {
            self.buf.pop_front();
            self.dropped += 1;
            dropped = 1;
        }
        self.buf.push_back(v);
        if dropped > 0 { Err(self.dropped) } else { Ok(()) }
    }

    pub fn lag(&self) -> u64 { self.dropped }
}
```

```python
# vad.py:105 — after: bounded, lag observable, never blocks the audio thread
self._input_ch = aio.BoundedRing[rtc.AudioFrame | VADStream._FlushSentinel](
    maxsize=50,                    # 50 x 20 ms = 1 s of audio
    on_drop=self._on_audio_dropped,
)
```

**Risk**: drop-oldest silently discards the oldest audio, so a slow provider loses the *start* of
the user's speech rather than the tail. The lag counter must therefore be a first-class
metric, and the drop policy must be a per-stage configuration, not a global default.

---

### refactor-2: One generic `ResilientProvider` instead of three `FallbackAdapter` copies

**File**: `livekit-agents/livekit/agents/stt/fallback_adapter.py:45-347` (540 lines),
`llm/fallback_adapter.py:38-408` (408 lines), `tts/fallback_adapter.py` (649 lines)

**Why it scales**: three near-identical state machines — status list, availability events,
recovery task per provider, sticky ordering, per-provider timeout/retry rewrite — with three
independent bug surfaces. A fix to "the primary is re-probed on every request" must be applied
three times, and it currently isn't.

**Before** (`livekit-agents/livekit/agents/llm/fallback_adapter.py:25-33` and `:96-112`):

```python
@dataclass
class _LLMStatus:
    available: bool
    recovering_task: asyncio.Task[None] | None
    selection_id: int = 0

class FallbackAdapter(LLM[Literal["llm_availability_changed"]]):
    def __init__(self, llm: list[LLM], *, attempt_timeout: float = 5.0, ...) -> None:
        self._status = [_LLMStatus(available=True, recovering_task=None) for _ in self._llm_instances]

    def _llm_order(self) -> list[int]:
        order = list(range(len(self._llm_instances)))
        if self._sticky:
            selected = max(order, key=lambda i: self._status[i].selection_id if ... else 0)
            order.remove(selected); order.insert(0, selected)
        return order

    def _next_instance(self) -> LLM:
        for i in self._llm_order():
            if self._status[i].available:
                return self._llm_instances[i]
        return self._llm_instances[0]
```

**After** (one generic implementation; STT/LLM/TTS supply only the typed parts):

```python
TProv = TypeVar("TProv")
TStream = TypeVar("TStream")

class ProviderStatus(Generic[TProv]):
    """Availability + recovery state for one provider in a resilient chain."""
    __slots__ = ("provider", "available", "recovering", "selection_id", "next_probe_at")
    def __init__(self, provider: TProv) -> None:
        self.provider = provider
        self.available = True
        self.recovering: asyncio.Task[None] | None = None
        self.selection_id = 0
        self.next_probe_at = 0.0     # monotonic: when to probe a failed provider again

class ResilientProvider(Generic[TProv, TStream]):
    """Generic failover chain.

    One implementation for STT, LLM and TTS. A failed provider is put on a
    backoff instead of being re-probed on every request, which is what the
    three hand-written adapters get wrong.
    """

    def __init__(
        self,
        providers: Sequence[TProv],
        *,
        attempt_timeout: float,
        max_retry_per_provider: int = 0,
        retry_interval: float = 0.5,
        probe_backoff: float = 30.0,
    ) -> None:
        self._status = [ProviderStatus(p) for p in providers]
        self._probe_backoff = probe_backoff
        self._active: TProv = providers[0]

    def _order(self) -> list[int]:
        return [i for i, s in enumerate(self._status) if s.available] or [0]

    def _mark_failed(self, i: int, *, now: float) -> None:
        s = self._status[i]
        s.available = False
        s.next_probe_at = now + self._probe_backoff      # <- do not re-probe every request
```

**Risk**: the three adapters are not actually behaviourally identical today (the STT one
replays the whole audio buffer to every recovery probe; the LLM one tracks `text_sent` to
decide whether a retry is safe). Collapsing them will surface those differences as bugs rather
than hide them — which is the point, but it means the collapse must land with per-provider
tests, not as a mechanical substitution.

---

### refactor-3: Replace the tee-based metrics fan-out with an event subscription

**File**: `livekit-agents/livekit/agents/stt/stt.py:459-463`, `llm/llm.py:256`,
`tts/tts.py:564-565`, `vad.py:108-112`; the fan-out primitive is
`utils/aio/itertools.py:20-66`

**Why it scales**: every provider event is written into *two* deques. The monitor peer is
consumed by a metrics task, but if that task is slow or abandoned, the second deque grows
without bound — a second unbounded queue on the same hot path as refactor-1.

**Before** (`livekit-agents/livekit/agents/stt/stt.py:457-463`):

```python
self._event_ch = aio.Chan[SpeechEvent]()

self._tee = aio.itertools.tee(self._event_ch, 2)
self._event_aiter, monitor_aiter = self._tee
self._metrics_task = asyncio.create_task(
    self._metrics_monitor_task(monitor_aiter), name="STT._metrics_task"
)
```

```python
# utils/aio/itertools.py:56-57 — every item is appended to every peer buffer
for peer_buffer in peers:
    peer_buffer.append(item)
```

**After** (the provider already has an event emitter; use it, and bound the subscriber):

```python
# stt/stt.py:457-463 — after
self._event_ch = aio.Chan[SpeechEvent](maxsize=256)
self._metrics_task = asyncio.create_task(
    self._metrics_monitor_task(self._event_aiter), name="STT._metrics_task"
)
# metrics are emitted on the provider's own emitter, not on a copy of the stream
self._stt.on("metrics_collected", self._on_metrics)      # STT already extends rtc.EventEmitter
```

```python
# a bounded subscriber ring for the one case that needs the stream, not the event
class SubscriberRing(Generic[T]):
    def __init__(self, cap: int = 256) -> None:
        self._buf: deque[T] = deque(maxlen=cap)
        self._waiter: asyncio.Future | None = None
        self._dropped = 0

    def offer(self, item: T) -> None:
        if len(self._buf) == self._buf.maxlen:
            self._buf.popleft(); self._dropped += 1
        self._buf.append(item)
        if self._waiter and not self._waiter.done():
            self._waiter.set_result(None)
```

**Risk**: the metrics monitor currently observes the *stream* of events, which is stronger than
observing the `metrics_collected` emitter (some events, e.g. `RECOGNITION_USAGE`, are only
visible on the stream). The move to an emitter subscription has to keep a stream-level hook for
those, or usage accounting silently stops.

---

### refactor-4: O(1) text accumulation instead of repeated string concatenation

**File**: `livekit-agents/livekit/agents/tts/tts.py:768` and `:786`

**Why it scales**: `+=` on a `str` copies the whole accumulated buffer on every token. A turn
that streams 2 000 tokens pays O(N²) bytes copied — ~2 MB of copying for a 1 KB answer, on the
critical path, before a single byte reaches the provider.

**Before** (`livekit-agents/livekit/agents/tts/tts.py:763-788`):

```python
def push_text(self, token: str) -> None:
    """Push some text to be synthesized"""
    if not token or self._input_ch.closed:
        return

    self._pushed_text += token          # O(len) copy per token

    if self._metrics_task is None:
        self._metrics_task = asyncio.create_task(...)

    if not self._mtc_text:
        if self._num_segments >= 1:
            logger.warning("... deprecated ...")
            return                      # <- token dropped, see finding F1

        self._num_segments += 1

    self._mtc_text += token             # O(len) copy per token
    self._input_ch.send_nowait(token)
    self._input_buffer.append(token)
```

**After**:

```python
def push_text(self, token: str) -> None:
    """Push some text to be synthesized."""
    if not token or self._input_ch.closed:
        return

    self._pushed_parts.append(token)     # O(1) amortized
    self._pushed_len += len(token)

    if self._metrics_task is None:
        self._metrics_task = asyncio.create_task(...)

    if not self._mtc_parts:
        if self._num_segments >= 1:
            # Do not silently drop the token: close the segment and start a new
            # stream, or raise. Dropping text the user will never hear is the
            # worst of the three options.
            raise RuntimeError(
                "SynthesizeStream handles one segment; create a new stream per segment"
            )
        self._num_segments += 1

    self._mtc_parts.append(token)
    self._input_ch.send_nowait(token)
```

```python
@property
def pushed_text(self) -> str:
    return "".join(self._pushed_parts)
```

**Risk**: `self._pushed_text` is read by the metrics path (`tts/tts.py:634`) and by the
`APIError` message at `:626`; both need to move to the joined property. The behavioural change
is the important half: turning a silent drop into a hard error can break a plugin that
currently relies on the drop.

---

### refactor-5: Bound the retry-replay buffer

**File**: `livekit-agents/livekit/agents/tts/tts.py:579` and `:673-678`

**Why it scales**: the entire input text of the stream is retained for the life of the stream
so it can be replayed on retry — but the buffer is only needed when `max_retry > 0`, and it is
never trimmed. A long turn with retries enabled holds the full transcript twice (once here,
once in `_pushed_text`).

**Before** (`livekit-agents/livekit/agents/tts/tts.py:578-580` and `:673-678`):

```python
# buffered input events for retry replay
self._input_buffer: list[str | SynthesizeStream._FlushSentinel] = []
self._input_ended = False
```

```python
# replay buffered input into a fresh channel for retry
self._input_ch = aio.Chan[str | SynthesizeStream._FlushSentinel]()
for event in self._input_buffer:
    self._input_ch.send_nowait(event)
if self._input_ended:
    self._input_ch.close()
```

**After**:

```python
# Only the current segment is replayable: a retry restarts the segment, not
# the turn. Older segments are already committed to the provider.
self._segment_buffer: list[str | SynthesizeStream._FlushSentinel] = []
self._retain_replay_buffer = self._conn_options.max_retry > 0
```

```python
if self._retain_replay_buffer:
    self._input_ch = aio.Chan[str | SynthesizeStream._FlushSentinel]()
    for event in self._segment_buffer:
        self._input_ch.send_nowait(event)
    if self._input_ended:
        self._input_ch.close()
```

**Risk**: the current code replays the *whole* input, not the current segment, so a retry after
segment 2 re-synthesizes segment 1 too. Restricting the replay to the current segment changes
what a retry produces — a behaviour change visible as duplicated speech if it is implemented
wrongly.

---

### refactor-6: Monotonic clock for every duration

**File**: `livekit-agents/livekit/agents/stt/fallback_adapter.py:273` and `:302`,
`llm/fallback_adapter.py:324` and `:403`, `tts/tts.py:690`, `stt/stt.py:126`

**Why it scales**: wall-clock arithmetic for elapsed time is wrong whenever NTP steps the
clock, and it is wrong silently — the error message says "failed after -3.000 seconds".

**Before** (`livekit-agents/livekit/agents/stt/fallback_adapter.py:273` and `:302-304`):

```python
async def _recognize_impl(self, buffer, *, language=NOT_GIVEN, conn_options) -> SpeechEvent:
    start_time = time.time()          # wall clock
    ...
    raise APIConnectionError(
        f"all STTs failed ({[stt.label for stt in self._stt_instances]}) after {time.time() - start_time} seconds"
    )
```

**After**:

```python
start = time.monotonic()              # monotonic: immune to clock steps
...
raise APIConnectionError(
    f"all STTs failed ({[stt.label for stt in self._stt_instances]}) "
    f"after {time.monotonic() - start:.3f}s"
)
```

**Risk**: `SpeechEvent.created_at` (`stt/stt.py:126`) and the `timestamp` fields on every
metrics model (`metrics/base.py`) are *wall clock on purpose* — they are meant to be comparable
with provider-side timestamps and with other machines. Only the *durations* become monotonic;
a blanket find-and-replace of `time.time()` would break cross-machine timestamp correlation.

---

### refactor-7: Split capability declaration from behaviour

**File**: `livekit-agents/livekit/agents/stt/stt.py:132-153` and `:328-336`,
`tts/tts.py:53-59` and `:230-235`, `stt/fallback_adapter.py:65-82`

**Why it scales**: `STTCapabilities(streaming=True)` is a free-form boolean that the framework
trusts when it decides how to schedule work, but nothing enforces that the plugin actually
implements `stream()`. The mismatch is discovered on the hot path, as a `NotImplementedError`
raised into the audio pipeline.

**Before** (`livekit-agents/livekit/agents/stt/stt.py:132-153` and `:328-336`):

```python
@dataclass
class STTCapabilities:
    streaming: bool
    interim_results: bool
    diarization: bool = False
    aligned_transcript: Literal["word", "chunk", False] = False
    offline_recognize: bool = True
    keyterms: bool = False
    chat_context: bool = False
    manual_flush: bool = False
```

```python
def stream(self, *, language=NOT_GIVEN, conn_options=...) -> RecognizeStream:
    raise NotImplementedError(
        "streaming is not supported by this STT, please use a different STT or use a StreamAdapter"
    )
```

**After** (two types; the capability is derived from the type, not declared separately):

```python
class STT(ABC, rtc.EventEmitter[...]):
    """Batch-capable speech-to-text provider."""

    def _recognize_impl(self, buffer, *, language, conn_options) -> SpeechEvent: ...

    @property
    def capabilities(self) -> STTCapabilities:
        return STTCapabilities(streaming=False, interim_results=False)


class StreamingSTT(STT):
    """An STT that can transcribe incrementally.

    Implementing `stream()` is the *only* way to get `capabilities.streaming == True`;
    the flag cannot drift from the behaviour.
    """

    @abstractmethod
    def stream(self, *, language=NOT_GIVEN, conn_options=...) -> RecognizeStream: ...

    @property
    def capabilities(self) -> STTCapabilities:
        caps = replace(super().capabilities, streaming=True)
        return caps
```

**Risk**: `FallbackAdapter` currently *upgrades* a batch STT to a streaming one by wrapping it
in `StreamAdapter` (`stt/fallback_adapter.py:76-82`). With the type split, that wrapping has to
produce a `StreamingSTT`, which means `StreamAdapter` must implement the streaming interface
rather than patching a capability flag — a wider change than it looks.

---

### refactor-8: A serializable error envelope

**File**: `livekit-agents/livekit/agents/stt/stt.py:156-162`, `tts/tts.py:69-75`,
`llm/llm.py:99`

**Why it scales**: the error object carried on the wire to traces and dashboards is dropped by
the serializer, so the only record of *what* failed is the log line. A provider error that
escapes the log (a deployment that ships logs off-host) leaves a trace with a category and no
cause.

**Before** (`livekit-agents/livekit/agents/stt/stt.py:156-162`):

```python
class STTError(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["stt_error"] = "stt_error"
    timestamp: float
    label: str
    error: Exception = Field(..., exclude=True)     # <- excluded from every dump
    recoverable: bool
```

**After**:

```python
@dataclass(frozen=True)
class ErrorEnvelope:
    """A serializable description of a provider failure."""

    kind: str                       # "timeout" | "connection" | "provider" | "cancelled"
    message: str
    exception_type: str             # fully-qualified class name
    traceback: str | None = None
    recoverable: bool = False

    @classmethod
    def from_exception(cls, exc: BaseException, *, recoverable: bool) -> "ErrorEnvelope":
        return cls(
            kind="timeout" if isinstance(exc, asyncio.TimeoutError) else "provider",
            message=str(exc),
            exception_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
            traceback="".join(traceback.format_exception(exc)),
            recoverable=recoverable,
        )
```

```python
class STTError(BaseModel):
    type: Literal["stt_error"] = "stt_error"
    timestamp: float
    label: str
    error: ErrorEnvelope            # now survives model_dump_json()
    recoverable: bool
```

**Risk**: the pydantic model is part of the public event payload; third-party listeners that read
`err.error` as an `Exception` break. A `__getattr__` shim on `ErrorEnvelope` that returns the
original exception where one is available softens this.

---

### refactor-9: Contain a plugin exception instead of killing the runtime

**File**: `ten-framework/core/src/ten_runtime/binding/python/interface/ten_runtime/async_extension.py:481-500`

**Why it scales**: one unhandled exception in one Python extension aborts the whole native
process. Every graph, every session, every other extension in that process dies with it. This
is the single most important thing the Rust core must *not* copy.

**Before** (`ten-framework/.../interface/ten_runtime/async_extension.py:481-500`):

```python
def _exit_on_exception(self, async_ten_env: AsyncTenEnv, e: BaseException):
    traceback_info = traceback.format_exc()

    err = async_ten_env.log(
        LogLevel.ERROR,
        f"Uncaught exception: {e} \ntraceback: {traceback_info}",
        option=LogOption(sync=True),
    )
    if err is not None:
        print(f"Uncaught exception: {e} \ntraceback: {traceback_info}")

    sys.stdout.flush()
    sys.stderr.flush()

    os._exit(1)          # <- the whole runtime dies
```

**After** (report the failure, mark the extension dead, keep the runtime alive):

```python
def _on_exception(self, async_ten_env: AsyncTenEnv, e: BaseException) -> None:
    """Contain a plugin failure.

    The extension is marked unusable and the error is reported to the runtime;
    other extensions and other sessions keep running. Only the *core* may
    decide to terminate the process.
    """
    self._failed = True
    async_ten_env.log(
        LogLevel.ERROR,
        f"Extension {self.name} raised; it will be disabled: {e}",
        option=LogOption(sync=True),
    )
    # Tell the runtime this extension is dead so the graph can reroute or
    # terminate the affected node without taking the process with it.
    async_ten_env.set_property_from_json(
        f"extensions.{self.name}.state", json.dumps({"status": "failed", "error": str(e)})
    )
```

**Risk**: this changes TEN's failure contract from "the process is gone" to "the extension is
gone". Anything that relies on the process exit as a crash signal (supervisors, health checks)
must be re-pointed at the per-extension state. For voicekit this is not optional: the design
constraint "plugins must fail without killing the session" *is* this refactor.

---

### refactor-10: Batch the Python-boundary audio hop

**File**: `ten-framework/core/src/ten_runtime/binding/python/interface/ten_runtime/async_extension.py:394-419`,
`ten-framework/core/src/ten_runtime/binding/python/native/extension/extension.c:322-358`

**Why it scales**: every audio frame crossing the boundary costs one `Future`, one coroutine
object, one event-loop wakeup, and a GIL acquire/release — per 10 ms of audio. The GIL is held
across the whole call into Python, so a busy Python extension blocks the native core.

**Before** (`ten-framework/.../interface/ten_runtime/async_extension.py:394-419`):

```python
@final
def _proxy_on_audio_frame(self, ten_env: TenEnv, audio_frame: AudioFrame) -> None:
    ...
    if is_single_thread_mode(ten_env):
        main_loop = self._global_thread_manager.get_thread()
        asyncio.run_coroutine_threadsafe(
            self._wrapper_on_audio_frame(self._async_ten_env, audio_frame),
            main_loop,
        )                       # one Future + one wakeup per frame
```

```c
/* ten-framework/core/src/ten_runtime/binding/python/native/extension/extension.c:334-358 */
PyGILState_STATE prev_state = ten_py_gil_state_ensure_internal();
...
py_audio_frame = ten_py_audio_frame_wrap(audio_frame);
PyObject_CallMethod((PyObject *)py_extension, "_proxy_on_audio_frame", "OO", ...);
ten_py_audio_frame_invalidate(py_audio_frame);     /* the buffer dies with the call */
ten_py_gil_state_release_internal(prev_state);
```

**After** (a bounded ring on the native side, one drain coroutine on the Python side, and the
GIL released before entering Python):

```rust
// native side: frames land in a bounded ring; Python drains in batches
pub struct ExtensionFrameInbox {
    ring: AudioRing<AudioFrame>,
}

impl ExtensionFrameInbox {
    /// Called from the audio thread. Never enters Python; never takes the GIL.
    pub fn offer(&mut self, frame: AudioFrame) -> Result<(), u64> {
        self.ring.push(frame)
    }
}
```

```python
# python side: one drain task, not one task per frame
async def _drain_audio_frames(self) -> None:
    while not self._stop.is_set():
        batch = self._inbox.drain(max_items=32, max_wait=0.02)
        if not batch:
            await asyncio.sleep(0)
            continue
        for frame in batch:
            await self.on_audio_frame(self._async_ten_env, frame)
```

```c
/* native side: release the GIL before calling into Python */
ten_py_gil_state_release_internal(prev_state);
PyObject_CallMethod(...);        /* Python runs without the core blocked */
ten_py_gil_state_ensure_internal();
```

And hand the PCM over without copying it. TEN already has this accessor and does not use it
on the audio path — the frame goes through the copying `bytearray` accessor
(`ten-framework/core/src/ten_runtime/binding/python/native/msg/audio_frame.c:188-205`) instead
of the locked-view one (`.../native/msg/audio_frame.c:124-146`):

```c
/* zero-copy: a locked memoryview of the frame's PCM buffer */
PyObject *ten_py_audio_frame_lock_buf(PyObject *self, ...) {
  ten_buf_t *data = ten_audio_frame_peek_buf(py_audio_frame->msg.c_msg);
  ten_msg_add_locked_res_buf(py_audio_frame->msg.c_msg, data->data, &err);
  return PyMemoryView_FromMemory((char *)data->data, (Py_ssize_t)data->size, PyBUF_WRITE);
}
```

**Risk**: batching adds up to one batch interval of latency (20 ms at the numbers above) to
audio delivery. That is acceptable for TTS egress but is exactly the latency a VAD cares
about, so the batch size has to be per-extension, not global. The zero-copy view adds a
second constraint: the buffer must stay locked for as long as Python holds the view, which
for a `bytearray` copy is instant but for a `memoryview` is until the view is released —
a leak here is a stalled audio source, so the view's lifetime must be owned by the extension,
not by Python's garbage collector.

---

### refactor-11: Split the `AgentActivity` god class

**File**: `livekit-agents/livekit/agents/voice/agent_activity.py:341` (5 532 lines, one class)

**Why it scales**: one class holds turn detection, endpointing, speech scheduling, tool
execution, the TTS task, the pipeline-reply task, the realtime task, interruption and
false-interruption recovery. A change to any of them risks the others, and there is no seam
at which a piece can be tested alone.

**Before** (`livekit-agents/livekit/agents/voice/agent_activity.py:341` — class declaration,
followed by ~90 methods spanning all of the above):

```python
class AgentActivity(RecognitionHooks):
    def __init__(self, agent: Agent, sess: AgentSession) -> None:
        ...
    # turn taking:      on_start_of_speech, on_end_of_speech, on_final_transcript,
    #                   on_eot_prediction, on_end_of_turn, _user_turn_completed_impl
    # scheduling:       _schedule_speech, _scheduling_task, wait_for_idle
    # speech:           say, _tts_task_impl, _pipeline_reply_task_impl,
    #                   _realtime_generation_task_impl
    # interruption:     interrupt, _interrupt_by_audio_activity,
    #                   _start_false_interruption_timer, _restore_paused_speech_state
    # tools:            _on_tool_execution_updated, _tool_items_added
```

**After** (four collaborators with narrow interfaces):

```python
class TurnController:            # start/end of user turn, endpointing decisions
    def on_start_of_speech(self, at: float) -> None: ...
    def on_end_of_speech(self, at: float) -> None: ...
    def on_final_transcript(self, text: str) -> None: ...
    def on_end_of_turn(self) -> bool: ...

class SpeechScheduler:           # priority queue of speech handles
    def schedule(self, handle: SpeechHandle, priority: int) -> None: ...
    def interrupt(self, *, force: bool) -> None: ...

class InterruptionController:    # barge-in and false-interruption recovery
    def begin(self) -> None: ...
    def restore(self, paused: _PausedSpeechInfo) -> None: ...

class ToolExecutor:              # tool calls, results, and their cancellation
    async def run(self, call: FunctionCall) -> ToolExecutionOutput: ...
```

```python
class AgentActivity(RecognitionHooks):
    def __init__(self, agent: Agent, sess: AgentSession) -> None:
        self._turn = TurnController(session=sess)
        self._scheduler = SpeechScheduler()
        self._interruptions = InterruptionController()
        self._tools = ToolExecutor()
```

**Risk**: `AgentActivity` holds a large amount of inter-state (the paused-speech snapshot, the
delegation check task, the away-timer hold counts). Splitting it without first writing down
the invariants will produce subtle ordering bugs that only appear under barge-in.

---

### refactor-12: Bound the VAD's accumulated utterance and make the batch adapter stream

**File**: `livekit-agents/livekit/agents/vad.py:46`, `stt/stream_adapter.py:112-164`

**Why it scales**: the VAD accumulates every frame of the user's utterance in a list and hands
the whole list to whoever asks. The `StreamAdapter` then merges those frames into one buffer
and issues a *single batch* recognize call — so a "streaming" STT that only supports batch
recognition still buffers the whole utterance before producing a single word.

**Before** (`livekit-agents/livekit/agents/vad.py:46-53`):

```python
frames: list[rtc.AudioFrame] = field(default_factory=list)
"""
List of audio frames associated with the speech.

- For `start_of_speech` events, this contains the audio chunks that triggered the detection.
- For `inference_done` events, this contains the audio chunks that were processed.
- For `end_of_speech` events, this contains the complete user speech.
"""
```

```python
# stt/stream_adapter.py:130-146
elif event.type == VADEventType.END_OF_SPEECH:
    ...
    merged_frames = utils.merge_frames(event.frames)
    t_event = await self._wrapped_stt.recognize(
        buffer=merged_frames, language=self._language, conn_options=self._wrapped_stt_conn_options,
    )
```

**After** (the VAD streams its frames out; the adapter feeds the wrapped provider
incrementally):

```python
@dataclass
class VADEvent:
    type: VADEventType
    samples_index: int
    timestamp: float
    speech_duration: float
    silence_duration: float
    frames: list[rtc.AudioFrame] = field(default_factory=list)
    # NEW: the frames of THIS segment only, released as they are consumed,
    # so a long turn does not hold the whole utterance in memory.
    stream: FrameStream | None = None
```

```python
# stt/stream_adapter.py — after: feed the wrapped provider incrementally
async def _recognize(self) -> None:
    async for event in vad_stream:
        if event.type == VADEventType.END_OF_SPEECH:
            # The batch provider still needs the whole utterance, but the
            # frames are now released as they are merged rather than held
            # in two places at once.
            merged = utils.merge_frames(event.frames)
            ...
```

**Risk**: `VADEvent.frames` is read by plugins (the Deepgram plugin reads it at
`livekit-plugins-deepgram/.../stt.py:616` region) and by the `StreamAdapter`; changing the
semantics of `frames` is a breaking change to the VAD plugin contract.

---

### refactor-13: Inject the consumer instead of a public mutable attribute

**File**: `vocode-core/vocode/streaming/transcriber/base_transcriber.py:19-25` and `:59`,
`vocode-core/vocode/streaming/streaming_conversation.py:618`

**Why it scales**: the transcriber's consumer is assigned from *outside* after construction, so
the provider cannot know at construction time whether it has a consumer. A provider that
produces a partial result before the consumer is wired drops it on the floor.

**Before** (`vocode-core/vocode/streaming/transcriber/base_transcriber.py:19-25` and `:55-59`):

```python
class AbstractTranscriber(Generic[TranscriberConfigType], AbstractWorker[bytes]):
    consumer: AbstractWorker[Transcription]          # public, unset at construction
    streaming_conversation: "StreamingConversation"

    def __init__(self, transcriber_config: TranscriberConfigType):
        AbstractWorker.__init__(self)
        self.transcriber_config = transcriber_config
        self.is_muted = False

    def produce_nonblocking(self, item: Transcription):
        self.consumer.consume_nonblocking(item)
```

```python
# streaming_conversation.py:616-618 — the wiring happens later, from outside
self.transcriptions_worker = self.TranscriptionsWorker(...)
self.transcriber.consumer = self.transcriptions_worker
```

**After** (the callback is a constructor argument; there is no unset window):

```python
class AbstractTranscriber(Generic[TranscriberConfigType], AbstractWorker[bytes]):
    def __init__(
        self,
        transcriber_config: TranscriberConfigType,
        *,
        on_transcription: Callable[[Transcription], None],
    ) -> None:
        super().__init__()
        self.transcriber_config = transcriber_config
        self._on_transcription = on_transcription

    def produce_nonblocking(self, item: Transcription) -> None:
        self._on_transcription(item)
```

**Risk**: this is a large API break for every Vocode transcriber. Vocode is stale, so the
break has no cost — which is precisely why it is the right shape to copy.

---

### refactor-14: O(1) eviction in the token-budget loop

**File**: `vocode-core/vocode/streaming/agent/openai_utils.py:139-149`,
`vocode-core/vocode/streaming/agent/token_utils.py:45-59`

**Why it scales**: eviction pops from the *front* of a list, so removing N messages is O(N²)
in list shifting; and the budget is a hard-coded model→context-size table that silently
mis-budgets every model released after it was written.

**Before** (`vocode-core/vocode/streaming/agent/openai_utils.py:135-149`):

```python
    num_removed_messages = 0
    while (
        context_size > get_chat_gpt_max_tokens(model_name) - LLM_AGENT_DEFAULT_MAX_TOKENS - 50
    ):  # context limit includes the max tokens, and 50 for safety
        if len(chat_messages) <= 1:
            logger.error(f"Prompt is too long to fit in context window, num tokens {context_size}")
            break
        num_removed_messages += 1
        chat_messages.pop(1)          # O(len) shift, repeated
```

**After** (a deque for O(1) eviction, and the budget supplied by the provider rather than a
table):

```python
from collections import deque

def fit_to_budget(
    messages: list[dict],
    *,
    context_limit: int,
    reserve: int,
) -> list[dict]:
    """Drop the oldest non-system messages until the prompt fits.

    ``context_limit`` comes from the provider's own model metadata, so a new
    model does not need a table edit.
    """
    head, tail = deque(), deque(messages)
    while tail and estimate_tokens(head) + estimate_tokens(tail) + reserve > context_limit:
        if len(tail) <= 1:
            break
        tail.popleft()
    return list(head) + list(tail)
```

**Risk**: the current code keeps the system preamble at index 0 by construction; a deque split
has to preserve that explicitly, and the token estimator is not free — the current code
computes the count once, the replacement must not re-tokenize the whole prompt per eviction.

---

### refactor-15: Bounded per-processor queues with an explicit overflow policy

**File**: `pipecat/src/pipecat/processors/frame_processor.py:140-191`,
`pipecat/src/pipecat/utils/frame_queue.py:33-44`,
`pipecat/src/pipecat/transports/base_input.py:318-322`

**Why it scales**: every processor owns two unbounded queues, and every transport owns a third.
A slow STT service therefore converts into unbounded memory growth upstream of it, not into a
visible lag.

**Before** (`pipecat/src/pipecat/processors/frame_processor.py:140-159`):

```python
class FrameProcessorQueue(asyncio.PriorityQueue):
    """A priority queue for the frames arriving at a frame processor.

    Frames are dequeued in three tiers: the `StartFrame` first, then
    `SystemFrame`, then data and control frames. Frames of the same tier keep
    their arrival order.
    """

    START_PRIORITY = 1
    SYSTEM_PRIORITY = 10
    DEFAULT_PRIORITY = 20

    def __init__(self):
        """Initialize the FrameProcessorQueue."""
        super().__init__()          # no maxsize -> unbounded
        self.__counter = 0
```

```python
# transports/base_input.py:318-322
def _create_audio_task(self):
    if not self._audio_task and self._params.audio_in_enabled:
        self._audio_in_queue = asyncio.Queue()     # no maxsize -> unbounded
        self._audio_task = self.create_task(self._audio_task_handler())
```

**After** (bounded, with the overflow policy stated in one place):

```python
class FrameProcessorQueue(asyncio.PriorityQueue):
    """A bounded priority queue for the frames arriving at a frame processor.

    Three tiers, same as before. The bound is explicit, and overflow drops
    the lowest-priority, oldest data frame — control and system frames are
    never dropped — while the drop is counted and reported.
    """

    START_PRIORITY = 1
    SYSTEM_PRIORITY = 10
    DEFAULT_PRIORITY = 20
    MAX_SIZE = 512

    def __init__(self, maxsize: int = MAX_SIZE):
        super().__init__(maxsize=maxsize)
        self.__counter = 0
        self.dropped_frames = 0

    async def put(self, item: tuple[Frame, FrameDirection, FrameCallback | None]):
        frame, _, _ = item
        if isinstance(frame, StartFrame):
            priority = self.START_PRIORITY
        elif isinstance(frame, SystemFrame):
            priority = self.SYSTEM_PRIORITY
        else:
            priority = self.DEFAULT_PRIORITY

        while self.full():
            # Drop the oldest data frame to make room; never drop control.
            self.dropped_frames += 1
            await self._drop_oldest_data_frame()

        self.__counter += 1
        await super().put((priority, self.__counter, item))
```

**Risk**: dropping audio frames changes the audio the user hears, and Pipecat's interruption
model already relies on frames surviving in queues until an interruption clears them. The
bound has to be large enough that normal operation never reaches it, and the drop counter has
to be a metric — otherwise a bound that is hit in production is invisible.

---

## Findings

24 findings. Tags are `[bug|perf|reliability|api|scale|security]`.

### F1 — Second-segment TTS text is silently dropped
**Tag**: `[bug]`
**Where**: `livekit-agents/livekit/agents/tts/tts.py:775-782`
`push_text` accumulates `self._pushed_text += token` at `:768` and *then* returns at `:782`
when `self._num_segments >= 1` — the token is added to the accumulated text but never
pushed to `self._input_ch` (`:787`). The provider never receives it, and the metrics text
disagrees with what was spoken.

### F2 — O(N²) string accumulation on the TTS hot path
**Tag**: `[perf]`
**Where**: `livekit-agents/livekit/agents/tts/tts.py:768` and `:786`
`self._pushed_text += token` and `self._mtc_text += token` each copy the whole accumulated
string per token. Same shape at `llm/chat_context.py` merge time (see F10).

### F3 — Every provider audio queue is unbounded
**Tag**: `[scale]`
**Where**: `livekit-agents/livekit/agents/utils/aio/channel.py:50-53` and `:154-158`
(`maxsize` defaults to `0`, and `full()` returns `False` for `maxsize <= 0`);
instantiated unbounded at `vad.py:105`, `stt/stt.py:456`, `tts/tts.py:562`, `llm/llm.py:254`.
There is no drop policy and no lag counter anywhere on the audio path.

### F4 — The metrics tee duplicates every event into a second unbounded deque
**Tag**: `[perf]`
**Where**: `livekit-agents/livekit/agents/utils/aio/itertools.py:56-57`; used at
`stt/stt.py:459`, `llm/llm.py:256`, `tts/tts.py:564`, `vad.py:108`. If the monitor peer is not
drained, the peer buffer grows without bound.

### F5 — The VAD holds the whole utterance in memory
**Tag**: `[scale]`
**Where**: `livekit-agents/livekit/agents/vad.py:46-53` (`frames: list[rtc.AudioFrame]`,
"the complete user speech" on `END_OF_SPEECH`), consumed by
`livekit-agents/livekit/agents/stt/stream_adapter.py:141` (`utils.merge_frames(event.frames)`),
which then issues one batch `recognize()` — so a batch-only STT still buffers the entire turn.

### F6 — Capability flags can drift from behaviour
**Tag**: `[api]`
**Where**: `livekit-agents/livekit/agents/stt/stt.py:132-153` vs `:328-336`;
`tts/tts.py:53-59` vs `:230-235`. `stream()` raises `NotImplementedError` on the hot path
while `capabilities.streaming` may be `True`; `stt/fallback_adapter.py:65-82` has to detect
the mismatch by attribute inspection and repair it with a `StreamAdapter`.

### F7 — Provider error payloads are dropped by the serializer
**Tag**: `[bug]`
**Where**: `livekit-agents/livekit/agents/stt/stt.py:161` and `tts/tts.py:74`
(`error: Exception = Field(..., exclude=True)`). The exception never appears in
`model_dump_json()`, so the trace payload records *that* a provider failed and not *why*.

### F8 — A dead provider is re-probed on every request
**Tag**: `[reliability]`
**Where**: `livekit-agents/livekit/agents/stt/fallback_adapter.py:275-304` — when all providers
are unavailable, `all_failed` forces the *primary* to be tried again on every request, and
`_try_recovery` (`:230-264`) is called on every failure. There is no backoff, so a permanently
dead provider costs a full timeout on every single turn.

### F9 — Recovery probes receive a full copy of the audio
**Tag**: `[perf]`
**Where**: `livekit-agents/livekit/agents/stt/fallback_adapter.py:382-404` — the forward task
pushes every frame to every entry in `self._recovering_streams` *and* to the main stream.
With one live probe the audio is copied twice per frame.

### F10 — `ChatContext.merge` is O(N·M)
**Tag**: `[perf]`
**Where**: `livekit-agents/livekit/agents/llm/chat_context.py:738-771` calls
`find_insertion_index` (`:905-916`, a reverse linear scan) and `list.insert` once per merged
item — a quadratic merge over a long conversation.

### F11 — Wall-clock arithmetic is used for elapsed time
**Tag**: `[bug]`
**Where**: `livekit-agents/livekit/agents/stt/fallback_adapter.py:273` and `:302`;
`llm/fallback_adapter.py:324` and `:403`. `time.time()` deltas go negative when the clock steps.
Pipecat hit exactly this and added a guard:
`pipecat/src/pipecat/processors/metrics/frame_processor_metrics.py:151-160` ("a wall clock
that stepped backwards mid measurement").

### F12 — `AgentActivity` is a 5 532-line god class
**Tag**: `[api]`
**Where**: `livekit-agents/livekit/agents/voice/agent_activity.py:341`. One class owns turn
detection, endpointing, speech scheduling, tools, the TTS/pipeline/realtime reply tasks,
interruption and false-interruption recovery. Nothing can be tested or replaced in isolation.

### F13 — The plugin contract carries no ABI version
**Tag**: `[api]`
**Where**: `livekit-agents/livekit/agents/plugin.py:18-28` versions the *plugin package*;
`stt/stt.py:168-181` and `tts/tts.py:81-97` version the *provider contract* not at all. A
plugin built against an older contract fails at call time, not at install time. TEN is closer
to the opposite: `packages/core_extensions/default_llm_extension_python/manifest.json:4`
carries a `"version": "0.11.75"` per extension — but it is metadata, not a runtime gate, so
*no* framework of the four versions the plugin ABI itself.

### F14 — A Python extension exception kills the whole runtime
**Tag**: `[reliability]`
**Where**: `ten-framework/core/src/ten_runtime/binding/python/interface/ten_runtime/async_extension.py:481-500`
— `_exit_on_exception` calls `os._exit(1)`. One bad extension takes down every graph, session
and extension in the process. This is the design constraint "plugins must fail without killing
the session", inverted.

The *sync* `Extension` path is the opposite failure, and just as damaging: the C boundary
prints the Python error and clears it
(`ten-framework/core/src/ten_runtime/binding/python/native/common/error.c:175-183`,
`PyErr_Clear()` at `:180`), and the call site asserts it did not happen
(`.../native/extension/extension.c:353-354`). Where asserts are compiled out, the exception
is discarded and the extension continues with a corrupted result; the error never reaches the
graph. voicekit needs neither behaviour: a contained error report naming the extension, the
callback and the payload.

### F15 — Per-frame GIL-held round trip at the language boundary
**Tag**: `[perf]`
**Where**: `ten-framework/.../interface/ten_runtime/async_extension.py:394-419` schedules one
coroutine per audio frame; `ten-framework/core/src/ten_runtime/binding/python/native/extension/extension.c:334-358`
acquires the GIL for the whole call into Python and releases it afterwards. One Future, one
wakeup and one GIL round trip per 10 ms of audio.

### F16 — One shared Python thread for every extension
**Tag**: `[scale]`
**Where**: `ten-framework/.../interface/ten_runtime/async_extension.py:162-227`
(`TEN_PYTHON_THREAD_MODE`, default `single_thread`) — all extensions share one asyncio loop on
one thread. The loop is owned by a process-wide singleton (`.../global_thread_manager.py:18`)
running as a single `PythonGlobalMainThread` (`:79-81`, uvloop at `:116`). A CPU-bound
extension (a local VAD, a local Whisper) blocks every other extension in the graph.
`multi_thread` mode exists but gives each extension its own thread and its own GIL contention.

### F17 — TEN's message path has no bound and no backpressure
**Tag**: `[scale]`
**Where**: `ten-framework/core/src/ten_runtime/engine/msg_interface/common.c:286-303`
(`ten_engine_append_to_in_msgs_queue` pushes onto an unbounded list under a mutex);
`ten-framework/core/src/ten_runtime/ten_env_proxy/internal/notify.c:154-163` (every notify is
a tail-posted runloop task). On the Python side the `Future` returned by
`asyncio.run_coroutine_threadsafe` is *discarded*
(`ten-framework/.../interface/ten_runtime/async_extension.py:406-410`), so the producer never
learns the consumer is behind — there is no backpressure signal at all, only unbounded
growth. The same defect appears per-extension in the AssemblyAI buffer
(`.../assemblyai_asr_python/audio_buffer_manager.py:24` and `:38-46`, an unbounded
`bytearray`). No drop policy and no lag counter anywhere on the path.

### F18 — TEN's provider contract lives in an unvendored package
**Tag**: `[api]`
**Where**: `ten-framework/packages/core_extensions/default_llm_extension_python/extension.py:7-17`
imports `ten_ai_base.llm2.AsyncLLM2BaseExtension`, and
`ai_agents/agents/ten_packages/extension/assemblyai_asr_python/extension.py:7-30` imports
`ten_ai_base.asr.*`. Neither package is in this tree, so a third-party extension author's
contract is pinned by a dependency the repository does not vendor or version-lock.

### F19 — The audio frame buffer is invalidated under the extension
**Tag**: `[bug]`
**Where**: `ten-framework/core/src/ten_runtime/binding/python/native/extension/extension.c:346-356`
— the frame is wrapped for Python and then `ten_py_audio_frame_invalidate(py_audio_frame)` is
called immediately after the Python call returns. A Python extension that retains the frame
object past the callback reads an invalidated buffer.

### F20 — Every pipeline and transport queue in Pipecat is unbounded
**Tag**: `[scale]`
**Where**: `pipecat/src/pipecat/processors/frame_processor.py:153` (`super().__init__()` on an
`asyncio.PriorityQueue`), `pipecat/src/pipecat/utils/frame_queue.py:36`,
`pipecat/src/pipecat/transports/base_input.py:321`. Only two `asyncio.Queue(maxsize=...)` calls
exist in the whole package, both in the Vonage client.

### F21 — The provider surface is one async generator
**Tag**: `[api]`
**Where**: `pipecat/src/pipecat/services/stt_service.py:555-556` (`run_stt`) and
`pipecat/src/pipecat/services/tts_service.py:820-821` (`run_tts`). This is the smallest
provider contract of the four frameworks, and the base class supplies resampling, metrics,
interruption handling and context plumbing — compare a LiveKit Deepgram STT plugin at
`livekit-plugins/livekit-plugins-deepgram/livekit/plugins/deepgram/stt.py` (1 162 lines) for
the same provider.

### F22 — Interruption handling scans the whole queue
**Tag**: `[reliability]`
**Where**: `pipecat/src/pipecat/processors/frame_processor.py:1161-1188` and
`pipecat/src/pipecat/utils/frame_queue.py:64-87` — `has_uninterruptible` is an O(n) scan of
every queued item, and `reset()` re-queues the kept items. Correct (it preserves
uninterruptible frames), but O(n) per interruption on a queue that is unbounded (F20).

### F23 — Provider failures in Vocode silently drop the utterance
**Tag**: `[reliability]`
**Where**: `vocode-core/vocode/streaming/utils/worker.py:206-228` — `InterruptibleWorker._run_loop`
catches `Exception`, logs it, and continues with the next item. The call survives, but the
utterance is never spoken and no error reaches the caller. Contrast
`vocode-core/vocode/streaming/agent/base_agent.py:359-369`, where the agent at least converts
the failure into "stop". *Stale codebase* — noted as a design reference.

### F24 — There is no way to cancel an in-flight extension callback
**Tag**: `[reliability]`
**Where**: `ten-framework/core/include_internal/ten_runtime/extension/extension.h:36`
(`TEN_EXTENSION_ON_XXX_WARNING_THRESHOLD_US 100000`) is the *only* response to a slow
callback: it logs a warning at
`ten-framework/core/src/ten_runtime/extension/extension.c:946-950` and then lets the callback
run to completion. Pending asyncio tasks are cancelled only at loop shutdown, with a 0.5 s
grace (`ten-framework/.../interface/ten_runtime/global_thread_manager.py:152-169`), and the
`_ten_stop_event` set by `_stop_thread`
(`ten-framework/.../interface/ten_runtime/async_extension.py:159-160`) stops the extension at
`on_deinit` without aborting a running `on_audio_frame`. A stuck plugin therefore cannot be
stopped, only waited on — barge-in *across* the language boundary is not expressible at all.

---

## Dimension × Framework matrix

Winner is stated for every dimension. Every framework is anchored on every dimension.

### 1. Transport & audio path — **Winner: Pipecat**

*Why*: Pipecat is the only one of the four with a typed frame hierarchy carrying an explicit
`interruptible` flag (`pipecat/src/pipecat/frames/frames.py:99-117`), priority tiers that
guarantee `StartFrame` and `SystemFrame` ordering
(`pipecat/src/pipecat/processors/frame_processor.py:140-191`), base-class resampling on both
ingress (`pipecat/src/pipecat/services/stt_service.py:278`) and egress
(`pipecat/src/pipecat/transports/base_output.py:798-807`), and a write timeout that escalates
to a permanent failure rather than hanging (`.../base_output.py:1064-1092`). LiveKit has the
better *egress* bound (`livekit-agents/livekit/agents/voice/room_io/_output.py:56`,
`rtc.AudioSource(..., queue_size_ms=200)`) but an unbounded ingress.
*Common defect*: all four are unbounded on ingress (F3, F20, F17) and none has a drop policy.

- LiveKit: `livekit-agents/livekit/agents/vad.py:105`; `.../voice/room_io/_output.py:56`
- Pipecat: `pipecat/src/pipecat/processors/frame_processor.py:140-191`;
  `pipecat/src/pipecat/transports/base_output.py:1064-1092`
- TEN: `ten-framework/.../interface/ten_runtime/async_extension.py:394-419`;
  `ten-framework/core/src/ten_runtime/engine/msg_interface/common.c:286-303`;
  `ten-framework/.../assemblyai_asr_python/audio_buffer_manager.py:24`
- Vocode: `vocode-core/vocode/streaming/streaming_conversation.py:815-816`;
  `vocode-core/vocode/streaming/telephony/constants.py:4-13` (*stale*)

### 2. Concurrency & isolation — **Winner: LiveKit**

*Why*: LiveKit has real process-level job isolation with a choice of executor per deployment
(`livekit-agents/livekit/agents/worker.py:135-137` — thread by default, process when the
process variant is installed), a warm process pool with idle-process accounting
(`.../ipc/proc_pool.py:32`, `:204-219`, `:307-319`), per-job context and log isolation
(`.../job.py:186-195`, `:295-331`), and a defined IPC protocol
(`.../ipc/proto.py:265-279`, `.../ipc/channel.py:34-60`). Pipecat isolates by worker/task but
shares one event loop; TEN isolates extensions by thread at best (F16); Vocode mixes threads
and asyncio with `janus` bridges (`vocode-core/vocode/streaming/utils/worker.py:76-107`).

- LiveKit: `livekit-agents/livekit/agents/worker.py:135-137`;
  `livekit-agents/livekit/agents/ipc/proc_pool.py:32`
- Pipecat: `pipecat/src/pipecat/pipeline/worker.py:1202-1206` (per-processor task);
  `pipecat/src/pipecat/pipeline/parallel_pipeline.py:25-80`
- TEN: `ten-framework/.../interface/ten_runtime/async_extension.py:162-227`;
  `ten-framework/.../interface/ten_runtime/global_thread_manager.py:18` (process-wide
  singleton), `:79-81` and `:116` (the single `PythonGlobalMainThread` loop);
  `ten-framework/core/src/ten_runtime/extension_thread/extension_thread.c:322-324` (one OS
  thread per extension *group*)
- Vocode: `vocode-core/vocode/streaming/utils/worker.py:76-107` (*stale*)

### 3. Turn-taking — **Winner: Pipecat**

*Why*: Pipecat makes turn-taking a *strategy list* with explicit decision modes
(`FIRST`/`ALL`, `pipecat/src/pipecat/turns/user_turn_strategies.py:51-93`), a controller that
refuses to end a turn while the user is audibly speaking
(`.../turns/user_turn_controller.py:676-703`), a silence watchdog with a scheduled-decision
escape hatch (`:705-723`), and interruption as a broadcast that preserves uninterruptible
frames (`.../processors/frame_processor.py:1048-1053`, `:1161-1188`). LiveKit's turn logic is
at least as capable — learned endpointing (`livekit-agents/livekit/agents/voice/endpointing.py:49-92`),
false-interruption recovery (`.../voice/agent_activity.py:5247-5270`) — but it is welded into
the god class (F12).

- LiveKit: `livekit-agents/livekit/agents/voice/endpointing.py:49-92`;
  `livekit-agents/livekit/agents/voice/agent_activity.py:2594-2635` (false-interruption timer)
- Pipecat: `pipecat/src/pipecat/turns/user_turn_controller.py:643-650` and `:676-723`;
  `pipecat/src/pipecat/turns/user_turn_strategies.py:51-93`
- TEN: `ten-framework/ai_agents/agents/ten_packages/extension/ten_turn_detection/turn_detector.py:20-24`
  and `:82-141` (the decision is an LLM call — `TurnDetectorDecision`); the turn logic lives in
  an extension, not in the runtime
- Vocode: `vocode-core/vocode/streaming/transcriber/deepgram_transcriber.py:313-330` (endpointing
  delegated to the STT provider) (*stale*)

### 4. Provider abstraction & plugin contract — **Winner: Pipecat** (smallest surface);
**TEN wins on versioning**

*Why*: Pipecat asks for exactly one method per provider kind — an async generator of frames
(`services/stt_service.py:555-556`, `services/tts_service.py:820-821`) — and the base class
owns everything else (F21). TEN's contract is the only one with a *versioned manifest*
(`packages/core_extensions/default_llm_extension_python/manifest.json`), but the Python
contract itself is versioned by an unvendored package (F18). LiveKit declares capabilities
explicitly but enforces them weakly (F6) and versions neither the contract nor the ABI (F13).
Vocode's contract is a public mutable attribute plus a side-channel callback (F13-style
coupling; see refactor-13).

- LiveKit: `livekit-agents/livekit/agents/stt/stt.py:132-153` and `:328-336`;
  `livekit-agents/livekit/agents/plugin.py:18-28`
- Pipecat: `pipecat/src/pipecat/services/stt_service.py:555-556`;
  `pipecat/src/pipecat/services/tts_service.py:820-821`
- TEN: `ten-framework/packages/core_extensions/default_llm_extension_python/manifest.json:4`;
  `ten-framework/.../interface/ten_runtime/addon_manager.py:100-143` (the
  `register_addon_as_extension` decorator — the actual registration contract);
  `ten-framework/.../interface/ten_runtime/async_extension.py:504-533`
- Vocode: `vocode-core/vocode/streaming/transcriber/base_transcriber.py:19-25` and `:55-59` (*stale*)

### 5. State & context management — **Winner: LiveKit**

*Why*: `ChatContext` is a typed item list with a real truncation that preserves the leading
instructions and the dynamic-instructions message
(`livekit-agents/livekit/agents/llm/chat_context.py:684-720`), a created-at-ordered merge
(`:738-771`), and a summarizer. Vocode's budget is a hard-coded model→size table
(`vocode-core/vocode/streaming/agent/token_utils.py:45-59`) with O(N²) eviction
(`.../openai_utils.py:139-149`) and an unbounded event log
(`.../streaming/models/transcript.py:119-125`). TEN's state is a graph-level property store,
not a per-conversation one (`ten-framework/.../interface/ten_runtime/ten_env.py:62-70`).
Pipecat has an `LLMContext` plus a summarizer
(`pipecat/src/pipecat/processors/aggregators/llm_context.py:113`,
`.../aggregators/llm_context_summarizer.py`) but the context lives in the aggregator pair
rather than in a first-class, versioned object.

- LiveKit: `livekit-agents/livekit/agents/llm/chat_context.py:684-720` and `:738-771`
- Pipecat: `pipecat/src/pipecat/processors/aggregators/llm_context.py:113`;
  `pipecat/src/pipecat/processors/aggregators/llm_context_summarizer.py`
- TEN: `ten-framework/.../interface/ten_runtime/ten_env.py:62-70`
- Vocode: `vocode-core/vocode/streaming/agent/openai_utils.py:135-149`;
  `vocode-core/vocode/streaming/models/transcript.py:119-125` (*stale*)

### 6. Failure handling — **Winner: Pipecat**

*Why*: Pipecat has an explicit error taxonomy (`ErrorCategory`), a policy for what happens when
a processor can no longer do its job (`ProcessorUnusablePolicy.CONTINUE / END / CANCEL` at
`pipecat/src/pipecat/pipeline/worker.py:147-159`), an `is_usable` state that handlers can read
(`.../processors/frame_processor.py:334`) and that `push_error_frame` can clear
(`.../processors/frame_processor.py:987-1033`), a permanent-failure escalation that replaces
the deprecated `fatal` flag, a consecutive-silent-context detector that detects a provider
that is "up" but not speaking (`.../services/tts_service.py:2399-2420`), and a reconnect loop
with a clean-close path (`.../services/websocket_service.py:489-517`). TEN's opposite is
`os._exit(1)` (F14); Vocode's is a silent drop (F23); LiveKit's error payload is not
serializable (F7) though its fallback adapters are the most sophisticated of the four.

- LiveKit: `livekit-agents/livekit/agents/stt/fallback_adapter.py:169-228` and `:266-304`;
  `livekit-agents/livekit/agents/stt/stt.py:156-162`
- Pipecat: `pipecat/src/pipecat/pipeline/worker.py:147-159`;
  `pipecat/src/pipecat/processors/frame_processor.py:987-1033`;
  `pipecat/src/pipecat/services/tts_service.py:2399-2420`
- TEN: `ten-framework/.../interface/ten_runtime/async_extension.py:481-500`;
  `ten-framework/.../interface/ten_runtime/error.py:20-45` (`TenErrorCode`);
  `ten-framework/core/src/ten_runtime/binding/python/native/common/error.c:175-183`
- Vocode: `vocode-core/vocode/streaming/utils/worker.py:206-228` (*stale*)

### 7. Observability — **Winner: LiveKit**

*Why*: LiveKit attributes latency per *turn stage* rather than per service:
`EOUMetrics` separates end-of-utterance delay, transcription delay and the user callback delay
(`livekit-agents/livekit/agents/metrics/base.py:131-149`); there are dedicated
`InterruptionMetrics` (`:219-234`), `VADMetrics` (`:121-128`) and `EOTInferenceMetrics`
(`:152-166`); spans follow the OpenTelemetry GenAI conventions
(`.../llm/llm.py:287-316`); and — uniquely among the four — the session detects a **blocked
event loop** and records count/total/max on the session span
(`livekit-agents/livekit/agents/voice/agent_session.py:2213-2234`). Pipecat has the better
*extension* point (a non-intrusive observer, `pipecat/src/pipecat/observers/base_observer.py:114-185`)
but less precise attribution. TEN has no telemetry surface in the runtime bindings (no
`telemetry`/`metrics` symbols under `core/src/ten_runtime/binding/python/`). Vocode has Sentry
spans (`vocode-core/vocode/utils/sentry_utils.py:25-40`) but `vocode/__init__.py:12` disables
the package's logger, so the logs are off by default (*stale*).

- LiveKit: `livekit-agents/livekit/agents/metrics/base.py:131-149`;
  `livekit-agents/livekit/agents/voice/agent_session.py:2213-2234`
- Pipecat: `pipecat/src/pipecat/observers/base_observer.py:114-185`;
  `pipecat/src/pipecat/processors/metrics/frame_processor_metrics.py:137-188`
- TEN: `ten-framework/.../interface/ten_runtime/async_extension.py:481-492` (the only error
  reporting path, and it exits the process);
  `ten-framework/core/include_internal/ten_runtime/extension/extension.h:36` (a slow-callback
  threshold — TEN's only runtime performance signal);
  `ten-framework/.../interface/ten_runtime/log_option.py`
- Vocode: `vocode-core/vocode/utils/sentry_utils.py:25-40`; `vocode-core/vocode/__init__.py:12` (*stale*)

### 8. Extensibility cost — **Winner: Pipecat**

*Why*: Pipecat's provider author writes one async generator and inherits resampling, metrics,
interruption handling, expression adapters and context plumbing
(`services/stt_service.py:555-556`, `services/tts_service.py:820-821`; the base class does the
rest at `services/tts_service.py:2282-2290`). LiveKit's Deepgram STT plugin is 1 162 lines for
one provider (`livekit-plugins/livekit-plugins-deepgram/livekit/plugins/deepgram/stt.py`), and
its fallback adapter alone is 540 lines of framework code the provider author must not
distinguish from their own. TEN's AssemblyAI ASR extension is ~1 183 non-test lines across six
files *plus* an unvendored base package (F18). Vocode's provider author writes a `_run_loop`
and reaches for a public `consumer` attribute (refactor-13) (*stale*).

- LiveKit: `livekit-plugins/livekit-plugins-deepgram/livekit/plugins/deepgram/stt.py:83-337`
  and `:480-877`
- Pipecat: `pipecat/src/pipecat/services/stt_service.py:555-556`;
  `pipecat/src/pipecat/services/cartesia/tts.py:829-857`
- TEN: `ten-framework/ai_agents/agents/ten_packages/extension/assemblyai_asr_python/extension.py:39-80`;
  `.../assemblyai_asr_python/recognition.py` (435 lines)
- Vocode: `vocode-core/vocode/streaming/transcriber/base_transcriber.py:19-62`;
  `vocode-core/vocode/streaming/synthesizer/base_synthesizer.py:350-393` (*stale*)

### 9. Deployment & scale — **Winner: LiveKit**

*Why*: LiveKit has a warm process pool with an explicit idle-process target and
spawn-on-demand, plus worker load reporting and drain
(`livekit-agents/livekit/agents/ipc/proc_pool.py:92-110`, `:204-219`, `:307-319`;
`.../worker.py:1377-1428`, `:969-999`), and a `SessionReport` for post-hoc analysis
(`.../voice/report.py:34-58`). Pipecat has a `WorkerRunner` with an in-process bus and
network buses (pgmq/Redis) for distributed workers
(`pipecat/src/pipecat/pipeline/worker.py:1426-1446`; bus at `pipecat/src/pipecat/bus/`), which
is a scale story for the *fleet*, not for a single session. TEN's graph runtime is
single-process by construction. Vocode has none (*stale*).

- LiveKit: `livekit-agents/livekit/agents/ipc/proc_pool.py:40-68` and `:204-209` (idle-process
  accounting and target), `:307-319` (spawn-to-target);
  `livekit-agents/livekit/agents/worker.py:155` and `:287-290` (`load_threshold`),
  `:1377-1390` (dispatch-latency accounting)
- Pipecat: `pipecat/src/pipecat/pipeline/worker.py:1426-1446`;
  `pipecat/src/pipecat/bus/network/` (pgmq/Redis buses)
- TEN: `ten-framework/core/src/ten_runtime/extension/extension.c:835-946` (per-callback
  timing); `ten-framework/core/src/ten_runtime/extension_thread/extension_thread.c:322-324`;
  `ten-framework/core/src/ten_runtime/engine/engine.c`
- Vocode: `vocode-core/vocode/streaming/streaming_conversation.py:693-736` (per-call start,
  no pool) (*stale*)

### 10. Language boundary — **Winner: TEN for the topology; LiveKit for the wire format**

*Why*: TEN is the only framework that *has* a native-core/plugin boundary: a C++ runtime with
a pybind11 binding and a Python extension ABI
(`ten-framework/core/src/ten_runtime/binding/python/native/extension/extension.c:322-358`,
`.../interface/ten_runtime/async_extension.py:504-533`). That proves the topology voicekit
wants. But TEN's implementation has the two defects that must not be copied: `os._exit(1)` on
a plugin exception (F14) and a GIL-held, one-Future-per-frame hop (F15). LiveKit's *wire
format* is the better model — a length-prefixed binary frame with a `MSG_ID`-keyed message
registry over a Unix socket (`livekit-agents/livekit/agents/ipc/channel.py:34-60`;
`.../ipc/proto.py:14-279`) — but it carries whole payloads (`InferenceRequest.data: bytes`,
`.../ipc/proto.py:171-187`), so it is a request/response protocol with no streaming message
types. Pipecat is pure Python, but its `WorkerBus` shows the shape of a network boundary
(`pipecat/src/pipecat/bus/messages.py`).

- LiveKit: `livekit-agents/livekit/agents/ipc/channel.py:34-43` (message dispatch) and
  `:62-71` (4-byte big-endian length prefix);
  `livekit-agents/livekit/agents/ipc/proto.py:171-212` and `:265-279`
- Pipecat: `pipecat/src/pipecat/bus/messages.py:32-117` (the `BusMessage` hierarchy);
  `pipecat/src/pipecat/bus/network/pgmq.py` and `.../bus/network/redis.py` (network buses)
- TEN: `ten-framework/core/src/ten_runtime/binding/python/native/extension/extension.c:322-358`;
  `ten-framework/.../interface/ten_runtime/async_extension.py:394-419` and `:481-500`;
  `ten-framework/core/src/ten_runtime/binding/python/native/msg/audio_frame.c:124-146` (a
  zero-copy `memoryview` path that exists) vs `:188-205` (the copying `bytearray` accessor
  the audio frame actually uses);
  `ten-framework/packages/core_addon_loaders/python_addon_loader/src/main.cc:527-541` (the
  loader hard-pins CPython 3.10)
- Vocode: none — Vocode is a single-language, single-process design; the only boundary is
  `janus.Queue` bridges between its worker threads and the asyncio loop
  (`vocode-core/vocode/streaming/utils/worker.py:76-107`) (*stale*)

---

## What this means for voicekit's Rust/Python boundary

Concrete consequences of the above, stated as decisions rather than aspirations.

**Topology (from TEN, minus its failure mode).** The Rust core owns the audio path, the turn
state machine and the session lifecycle. Python plugins are *providers* (STT/LLM/TTS/VAD),
*tools*, and *user agent logic*. The boundary is a length-prefixed binary protocol over a Unix
socket, modelled on `livekit-agents/livekit/agents/ipc/channel.py:34-43` and `:62-71` — but with
streaming message types, which LiveKit's protocol lacks.

Two things to copy from TEN and one to avoid. *Copy the registration contract*: a decorator
plus a manifest that names the extension and its version
(`ten-framework/.../interface/ten_runtime/addon_manager.py:100-143`,
`.../default_llm_extension_python/manifest.json:4`). *Copy the zero-copy frame transport*:
TEN already has an accessor that hands Python a locked `memoryview` of the PCM buffer
(`ten-framework/core/src/ten_runtime/binding/python/native/msg/audio_frame.c:124-146`), but its
audio-frame path uses the copying `bytearray` accessor instead
(`.../native/msg/audio_frame.c:188-205`) — voicekit should expose the view and let Python read
the PCM without a copy. *Avoid the interpreter pin*: TEN's Python addon loader dlopens a
specific CPython 3.10 (`ten-framework/packages/core_addon_loaders/python_addon_loader/src/main.cc:527-541`),
which hard-fails on any other interpreter. A Rust/Python boundary must negotiate the ABI at
runtime, not pin the interpreter version at the loader.

**Message types the boundary must carry** (each a distinct frame, each independently
cancellable):

| Message | Direction | Carries |
|---|---|---|
| `AudioIn` | core → plugin | PCM chunk, sample rate, channels, capture timestamp |
| `AudioOut` | plugin → core | PCM chunk, segment id, is-final |
| `Transcript` | plugin → core | text, is-final, confidence, start/end time |
| `TextDelta` | plugin → core | LLM token delta, tool-call delta |
| `ToolCall` / `ToolResult` | both | call id, name, arguments, result |
| `Error` | both | `ErrorEnvelope` (F7/refactor-8): kind, message, exception type, traceback, recoverable |
| `Cancel` | both | target message id / stream id — the cancellation primitive that TEN lacks entirely |
| `EndOfTurn` / `Interruption` | core → plugin | the turn boundary and barge-in signal |
| `Capabilities` | plugin → core | declared once at attach time (refactor-7): streaming, interim results, aligned transcript |

**Cancellation.** Today TEN offers no way to cancel an in-flight extension callback or an
in-flight audio frame. The only response to a slow callback is a warning log above a 100 ms
threshold (`ten-framework/core/include_internal/ten_runtime/extension/extension.h:36`,
`.../core/src/ten_runtime/extension/extension.c:946-950`); pending asyncio tasks are
cancelled only at loop shutdown, with a 0.5 s grace
(`ten-framework/.../interface/ten_runtime/global_thread_manager.py:152-169`); and
`_ten_stop_event` (`.../interface/ten_runtime/async_extension.py:159-160`) stops the extension
at `on_deinit` without aborting a running `on_audio_frame`. Pipecat's cancellation is the most
complete of the four — it cancels the processor task and then *selectively preserves*
uninterruptible frames (`pipecat/src/pipecat/processors/frame_processor.py:1161-1188`,
`.../utils/frame_queue.py:75-82`). voicekit should adopt Pipecat's selective-drop semantics
and expose them across the boundary as an explicit `Cancel` message, because a plugin that is
mid-`run_tts` must be told to stop *and* must be told which of its in-flight messages are
still wanted.

**Error propagation.** Rules derived from the evidence: (a) a plugin exception must never
terminate the core — refactor-9; (b) the error payload must be serializable — refactor-8;
(c) a provider that is up but not producing must be detected — Pipecat's consecutive-silent-
context detector (`services/tts_service.py:2399-2420`); (d) a failed provider must not be
re-probed on every request — F8; (e) durations are monotonic — refactor-6.

**Extensibility.** One manifest, one small interface, no core edits — the Pipecat shape
(F21) with TEN's manifest versioning (F13): a plugin declares `kind`, `version`, `abi`, and
its capabilities; the core validates `abi` at attach time rather than failing at call time.

**The audio path.** Bounded rings with drop-oldest and a lag counter on both sides of the
boundary (refactor-1), because all four frameworks get this wrong (F3, F17, F20) and the
design constraint already names it.

---

## Priority table

| # | Refactor | File | Pattern eliminated | Throughput lever | Impl effort |
|---|---------|------|--------------------|------------------|-------------|
| 1 | Bounded audio ring + lag counter | `livekit-agents/livekit/agents/vad.py:105` | unbounded queue on the audio hot path | bounds memory; turns stall into a countable lag | S (core-side) |
| 2 | Contain plugin exceptions | `ten-framework/.../interface/ten_runtime/async_extension.py:481` | `os._exit(1)` from a plugin callback | one bad plugin no longer kills every session | S |
| 3 | Batch the language-boundary audio hop | `ten-framework/.../native/extension/extension.c:334` | one Future + one GIL round trip per frame | ~10x fewer boundary crossings per second of audio | M |
| 4 | Generic `ResilientProvider` | `livekit-agents/livekit/agents/stt/fallback_adapter.py:45` | three 400–650-line copies of one state machine | one bug fix instead of three; adds probe backoff | M |
| 5 | Event subscription instead of tee | `livekit-agents/livekit/agents/stt/stt.py:459` | second unbounded deque per provider event | removes a full copy of every event | S |
| 6 | O(1) text accumulation | `livekit-agents/livekit/agents/tts/tts.py:768` | quadratic string copying per token | removes O(N²) bytes copied per turn | S |
| 7 | Bounded retry-replay buffer | `livekit-agents/livekit/agents/tts/tts.py:579` | unbounded transcript retention | halves peak transcript memory | S |
| 8 | Monotonic durations | `livekit-agents/livekit/agents/stt/fallback_adapter.py:273` | wall-clock arithmetic for elapsed time | correct under clock steps | S |
| 9 | Serializable error envelope | `livekit-agents/livekit/agents/stt/stt.py:161` | exception dropped by the serializer | errors become visible in traces | S |
| 10 | Capability derived from type | `livekit-agents/livekit/agents/stt/stt.py:328` | flag/behaviour drift; runtime `NotImplementedError` | failures move to install time | M |
| 11 | Split `AgentActivity` | `livekit-agents/livekit/agents/voice/agent_activity.py:341` | 5 532-line god class | enables per-stage testing and replacement | L |
| 12 | Bound the VAD utterance buffer | `livekit-agents/livekit/agents/vad.py:46` | whole-utterance frame list + batch merge | removes one full copy of the turn's audio | M |
| 13 | Injected consumer callback | `vocode-core/vocode/streaming/transcriber/base_transcriber.py:19` | public mutable `consumer` attribute set post-construction | removes an unset window that drops partials | S |
| 14 | O(1) token-budget eviction | `vocode-core/vocode/streaming/agent/openai_utils.py:139` | quadratic list shifting; stale hard-coded budget table | O(N) eviction | S |
| 15 | Bounded processor queues | `pipecat/src/pipecat/processors/frame_processor.py:153` | unbounded queues everywhere in the pipeline | bounds memory under provider stall | M |

---

*No file under any `reference/` tree, under `livekit/agents`, or under the run directory other
than this report was created, modified, deleted, or executed. All findings are anchored to
`file:line` in the core runtime packages of the four targets.*
