"""The sidecar adapter conformance kit.

Every provider that plugs into the sidecar must obey the same seven
invariants. This module is the single, provider-neutral suite that proves it.
It is a library, not a test: it imports no test framework, opens no socket,
writes no file, and never shells out. A third-party adapter author can import
it from their own tests and call :func:`run` against their provider.

The contract
------------
``run(target, *, fixtures_dir)`` returns exactly one
:class:`ConformanceResult` per entry in :data:`CONFORMANCE_INVARIANTS`, in that
order. A failing invariant is a result with ``passed=False`` and a ``detail``
string that names the observation (counts, classifications, branch names) but
never a credential. ``run`` does not raise for a failed probe; it raises
:class:`ConformanceTargetError` when the *target* is not one of the accepted
forms, and :class:`~parlevox_sdk.adapters.base.AdapterConfigError` (its base
class) when the fixture directory is missing or the modality is unrecognised.
Either way the refusal happens before any probe runs: the target is normalised
first, then the fixture directory is opened by the kit's replay-only transport,
and only then is the probe list built.

``target`` must be **already constructed**, in one of exactly two forms:

* a :class:`~parlevox_sdk.plugin.Plugin` instance — its ``stt``/``llm``/``tts``/
  ``vad`` slots are inspected, and the first real (non-fake) adapter is the one
  under test;
* an adapter instance — wrapped in a ``Plugin`` so the modalities the adapter
  does not implement are exercised against the bundled fakes.

A class is refused. The kit never constructs the caller's target instance:
doing so would have to pick a transport, and only the caller knows which one is
offline. Every real adapter in the target must be wired to a
:class:`~parlevox_sdk.adapters.base.FixtureTransport` — its constructor must
accept ``transport`` and its ``_transport`` must already be a
``FixtureTransport``. Anything else (a class, an adapter on ``HttpTransport``,
a transport-less provider that could dial) raises
:class:`ConformanceTargetError`, whose message names the accepted forms. The
bundled fakes (``FakeStt``/``FakeLlm``/``FakeTts``/``FakeVad``) are exempt:
they are the offline baseline and hold no transport.

Probes still need extra provider instances (a fresh replay over the same
cassette, an empty-body transport, a pre-cancelled token). Each is built from
the target's own class over a kit-owned ``FixtureTransport``, so a provider must
be constructible more than once and must not depend on a key being present at
import time. A class that does not accept ``transport`` is a configuration
error rather than being constructed blindly. ``fixtures_dir`` is the vendor
cassette directory the provider's own tests use; the provider must resolve
every request against it, and the directory is required to exist before any
probe runs.

The drop-counter probe is host-owned rather than adapter-owned: the kit builds
its own plugin, so no real adapter, adapter class or adapter transport is ever
consulted and no credential can be resolved. When the target is a ``Plugin``
subclass whose constructor is still exactly ``Plugin.__init__``, a fresh
instance of that class is built — still from nothing but the bundled fakes, so
an under-reporting ``_send`` is exercised; every other target gets a plain
``Plugin()``. The four slots are verified to be the exact bundled fakes before a
frame is driven. A failure names counts and wire classes only — never frame
text, a provider message or a credential.

The seven invariants
--------------------

``stt-partials-then-one-final``
    ``transcribe(pcm)`` yields zero or more partial chunks (``final: False``)
    followed by **exactly one** final chunk (``final: True``) which is the last
    one. Checked by draining the generator over 160 samples of silence and
    counting finals and their position.

``tts-empty-text-refused``
    ``synthesize(text)`` rejects ``""`` and whitespace-only text. Checked by
    draining the generator for both inputs and requiring an exception
    (``ValueError`` for a caller defect, ``AdapterError`` for a classified
    one) **before a single audio frame is yielded**.

``tts-no-unrequested-audio``
    A request that produces no audio must produce no frames. Checked against a
    synthetic 200 response with an empty body: a correct adapter yields zero
    frames for non-empty text. A provider that accepts no transport at all
    (the bundled fakes) cannot replay an empty body; for those the probe uses
    a documented deterministic branch instead — two calls with the same text
    must yield the same, non-zero, number of frames, proving the provider does
    not invent audio per call.

``llm-structured-history-accepted``
    ``generate(prompt, history=...)`` accepts an oldest-first sequence of
    ``{"role": str, "text": str}`` turns and yields at least one delta. The
    signature is inspected first, so a one-argument ``generate(self, prompt)``
    provider fails the invariant rather than silently losing context.

``cancellation-propagates``
    A pre-cancelled ``CancelToken`` handed to a constructor that accepts
    ``cancel`` must surface as an ``AdapterError`` classified ``cancelled``
    before any work is emitted. A provider with no cooperative token is checked
    down the host-owned path instead: the plugin must acknowledge the cancel
    and suppress the request's work (no ``final`` and no ``done``), which is
    the tombstone branch reported in the detail.

``errors-classified``
    A wire failure must reach the caller as a classified ``AdapterError``.
    Checked with a synthetic catch-all transport that answers every request
    with HTTP 500: the resulting error's ``wire_class()`` must be one of
    ``transient``/``fatal``/``protocol``/``cancelled``/``capacity``. A
    transport-less provider is checked through the plugin instead: a
    ``ValueError`` it raises is mapped to the ``protocol`` wire class exactly
    as ``plugin.py`` maps it.

``drop-counters-honest``
    The host owns the bounded pending-audio buffer. Driving a host-built plugin
    with ``MAX_PENDING_AUDIO_FRAMES + 3`` frames and then ``audio_end`` must
    report ``dropped_frames == 3`` on **exactly one** terminal ``done`` message
    and emit no ``error`` frame. The plugin is ``Plugin()`` (bundled fakes, no
    transport, no credential), or — when the target is a ``Plugin`` subclass
    whose constructor is still exactly ``Plugin.__init__`` — a fresh instance of
    that class, still built from the bundled fakes. No real adapter, adapter
    class or adapter transport is touched, yet the plugin's own send behaviour
    — including an under-reporting ``_send`` — is still exercised. An adapter
    cannot violate this invariant; a plugin that under-reports its drops can.
    When the counter cannot be read, the detail names the cause with counts and
    wire classes only.

Offline guarantee
-----------------
Every probe is replay-only. :class:`~parlevox_sdk.adapters.base.FixtureTransport`
is the only transport this module ever constructs, and every real adapter in
the target must already carry one. A target that is a class, a provider whose
constructor takes no ``transport``, or a provider whose ``_transport`` is
anything other than a ``FixtureTransport`` is refused before a single probe
runs, so calling :func:`run` cannot dial out even if a provider's base URL
points at a live host. No code path constructs an ``HttpTransport``, opens a
socket or opens a cassette for writing.
"""

from __future__ import annotations

import inspect
import io
from dataclasses import dataclass

from .adapters.base import (
    AdapterConfigError,
    AdapterError,
    CancelToken,
    ErrorClass,
    FixtureTransport,
    redact,
)
from .fakes import FakeLlm, FakeStt, FakeTts, FakeVad
from .plugin import (
    MAX_PENDING_AUDIO_FRAMES,
    CancelledTombstones,
    Plugin,
)
from .protocol import KIND_JSON, decode_json, read_frame

__all__ = [
    "CONFORMANCE_INVARIANTS",
    "ConformanceResult",
    "ConformanceTargetError",
    "run",
]

#: The stable, ordered invariant ids this kit checks, one per red line.
CONFORMANCE_INVARIANTS = (
    "stt-partials-then-one-final",
    "tts-empty-text-refused",
    "tts-no-unrequested-audio",
    "llm-structured-history-accepted",
    "cancellation-propagates",
    "errors-classified",
    "drop-counters-honest",
)

#: The credential handed to a provider reconstructed by a probe. It is a
#: placeholder, never a real key, and is shaped so ``redact`` scrubs it if it
#: ever reached a message. It carries the word ``test`` so it is an obviously
#: fake fixture value rather than a credential-shaped secret.
PLACEHOLDER_KEY = "sk-test-conformance-placeholder-0000"

#: Modality -> bundled fake, used to fill the slots a target does not provide.
_FAKE_CLASSES = {
    "stt": FakeStt,
    "llm": FakeLlm,
    "tts": FakeTts,
    "vad": FakeVad,
}
_STOCK_FAKES = tuple(_FAKE_CLASSES.values())
_MODALITIES = tuple(_FAKE_CLASSES)

#: Wire classes the sidecar accepts for a classified failure.
_WIRE_CLASSES = frozenset(
    {"transient", "fatal", "protocol", "cancelled", "capacity"}
)

#: 160 mono samples of silence: enough that every STT adapter accepts it.
_PROBE_PCM = b"\x00\x00" * 160
_PROBE_TEXT = "conformance probe"
_PROBE_PROMPT = "conformance probe"
_PROBE_HISTORY = ({"role": "user", "text": "earlier turn"},)

#: Frames driven into the bounded pending-audio buffer for the drop probe.
_AUDIO_PROBE_FRAMES = MAX_PENDING_AUDIO_FRAMES + 3
_EXPECTED_DROPS = 3

_CANCEL_ID = "conformance-cancel"
_ERROR_ID = "conformance-error"
_DROP_ID = "conformance-drop"


@dataclass(frozen=True)
class ConformanceResult:
    """One invariant's verdict.

    ``detail`` names what was observed — frame counts, chunk counts, the error
    classification, or the branch the probe took. It never carries a credential
    or a provider's raw exception text; every detail is passed through
    :func:`~parlevox_sdk.adapters.base.redact` as defence in depth.
    """

    invariant: str
    passed: bool
    detail: str


class ConformanceTargetError(AdapterConfigError):
    """The target is not an accepted conformance form.

    Raised by :func:`run` before any probe when ``target`` is a class, a
    provider that takes no ``transport``, or a provider not wired to a
    :class:`~parlevox_sdk.adapters.base.FixtureTransport`. Its message names
    both accepted forms — a ``Plugin`` instance or an adapter instance built
    over a ``FixtureTransport`` — and no credential. It is an
    :class:`~parlevox_sdk.adapters.base.AdapterConfigError`, so callers that
    already treat target defects as configuration failures keep working.
    """

    def __init__(self, provider, message):
        super().__init__(provider, message)


class _Subject:
    """A normalised target: the plugin to drive plus the adapter under test."""

    __slots__ = ("plugin", "primary", "modality", "is_plugin_target", "plugin_cls")

    def __init__(self, plugin, primary, modality, is_plugin_target, plugin_cls=None):
        self.plugin = plugin
        self.primary = primary
        self.modality = modality
        self.is_plugin_target = is_plugin_target
        # Recorded at normalisation so the drop-counter probe can reason about
        # the target's plugin class without reading the live instance. ``None``
        # derives it from the instance.
        self.plugin_cls = type(plugin) if plugin_cls is None else plugin_cls


# -- construction helpers ------------------------------------------------


def _constructor_params(cls):
    """The ``__init__`` parameters of ``cls``, or an empty mapping."""
    try:
        return inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return {}


def _callable_params(function):
    """The parameters of a bound callable, or an empty mapping."""
    try:
        return inspect.signature(function).parameters
    except (TypeError, ValueError):
        return {}


def _accepts(parameters, name):
    """Whether a signature accepts ``name`` positionally, by keyword or ``**``."""
    parameter = parameters.get(name)
    if parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        return True
    return any(
        item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()
    )


def _construct_adapter(cls, transport, *, cancel=None):
    """Construct ``cls`` with only the keywords its signature accepts."""
    parameters = _constructor_params(cls)
    kwargs = {}
    if transport is not None and _accepts(parameters, "transport"):
        kwargs["transport"] = transport
    if _accepts(parameters, "environ"):
        env_var = getattr(cls, "ENV_VAR", "")
        kwargs["environ"] = {env_var: PLACEHOLDER_KEY} if env_var else {}
    if cancel is not None and _accepts(parameters, "cancel"):
        kwargs["cancel"] = cancel
    return cls(**kwargs)


def _fixture_transport(fixtures_dir):
    """A fresh replay-only transport over ``fixtures_dir``."""
    return FixtureTransport.from_directory(
        fixtures_dir,
        provider="conformance",
        max_attempts=1,
        sleeper=lambda _seconds: None,
    )


def _fresh_adapter(cls, fixtures_dir):
    """Construct ``cls`` over a fresh fixture transport.

    A class that does not accept ``transport`` cannot be replayed offline, so it
    is a configuration error; the kit never constructs a socket-capable
    provider in its place.
    """
    if not _accepts(_constructor_params(cls), "transport"):
        raise AdapterConfigError(
            "conformance",
            f"{cls.__name__} does not accept a transport; the kit only "
            "constructs providers over a FixtureTransport",
        )
    return _construct_adapter(cls, _fixture_transport(fixtures_dir))


def _empty_body_transport():
    """A transport whose every endpoint returns 200 with an empty body."""
    return FixtureTransport(
        [
            {
                "method": "POST",
                "path": "",
                "responses": [
                    {
                        "status": 200,
                        "headers": {"content-type": "audio/pcm"},
                        "chunks": [],
                    }
                ],
            }
        ],
        provider="conformance",
        max_attempts=1,
        sleeper=lambda _seconds: None,
    )


def _catch_all_transport():
    """A transport whose every endpoint answers HTTP 500."""
    spec = {
        "status": 500,
        "headers": {"content-type": "application/json"},
        "chunks": [],
    }
    return FixtureTransport(
        [
            {"method": "POST", "path": "", "responses": [dict(spec)]},
            {"method": "GET", "path": "", "responses": [dict(spec)]},
        ],
        provider="conformance",
        max_attempts=1,
        sleeper=lambda _seconds: None,
    )


# -- normalisation -------------------------------------------------------


def _modality_of(provider):
    """The manifest modality of an adapter instance, or a config error."""
    manifest = getattr(provider, "manifest", None)
    modality = manifest.get("modality") if isinstance(manifest, dict) else None
    if modality not in _FAKE_CLASSES:
        raise AdapterConfigError(
            "conformance",
            f"target does not expose a recognised modality: {type(provider).__name__}",
        )
    return modality


def _refuse_non_fixture(provider):
    """Refuse a real provider that could reach a non-replay transport.

    This is the socket guarantee, and it runs before any probe. The bundled
    fakes are exempt — they are the offline baseline and hold no transport. Any
    other provider must accept a ``transport`` constructor keyword and already
    hold a :class:`FixtureTransport`; a transport-less dialer and an
    ``HttpTransport``-backed instance are both refused without a connection
    attempt.
    """
    # Exact-type match: a subclass of a fake is not the bundled baseline and is
    # held to the same transport rules as any other provider.
    if type(provider) in _STOCK_FAKES:
        return
    cls = type(provider)
    if not _accepts(_constructor_params(cls), "transport"):
        raise ConformanceTargetError(
            "conformance",
            f"{cls.__name__} takes no transport keyword; pass a Plugin "
            "instance or an adapter instance built over a FixtureTransport",
        )
    transport = getattr(provider, "_transport", None)
    if not isinstance(transport, FixtureTransport):
        raise ConformanceTargetError(
            "conformance",
            f"{cls.__name__} is not wired to a FixtureTransport "
            f"({type(transport).__name__}); pass a Plugin instance or an "
            "adapter instance built over a FixtureTransport",
        )


def _first_real_adapter(plugin):
    """The first non-fake adapter in a plugin, in modality order, or ``None``."""
    for modality in _MODALITIES:
        adapter = getattr(plugin, modality, None)
        if adapter is not None and not isinstance(adapter, _STOCK_FAKES):
            return adapter
    return None


def _normalise(target, fixtures_dir):
    """Turn a plugin or adapter instance into a subject; refuse anything else."""
    if isinstance(target, type):
        raise ConformanceTargetError(
            "conformance",
            "conformance targets must be already-constructed instances: pass a "
            "Plugin instance or an adapter instance built over a "
            "FixtureTransport, not a class",
        )

    if isinstance(target, Plugin):
        for modality in _MODALITIES:
            adapter = getattr(target, modality, None)
            if adapter is not None:
                _refuse_non_fixture(adapter)
        primary = _first_real_adapter(target)
        modality = _modality_of(primary) if primary is not None else None
        return _Subject(target, primary, modality, True, type(target))

    _refuse_non_fixture(target)
    modality = _modality_of(target)
    return _Subject(Plugin(**{modality: target}), target, modality, False, Plugin)


def _provider_for(subject, modality, fixtures_dir):
    """A fresh provider for ``modality`` that keeps the target's own class."""
    if modality not in _FAKE_CLASSES:
        raise AdapterConfigError(
            "conformance", f"unknown modality {modality!r}"
        )
    if subject.is_plugin_target:
        adapter = getattr(subject.plugin, modality, None)
        if adapter is None or isinstance(adapter, _STOCK_FAKES):
            return _FAKE_CLASSES[modality]()
        return _fresh_adapter(type(adapter), fixtures_dir)
    if modality == subject.modality:
        if type(subject.primary) in _STOCK_FAKES:
            return _FAKE_CLASSES[modality]()
        return _fresh_adapter(type(subject.primary), fixtures_dir)
    return _FAKE_CLASSES[modality]()


# -- shared message/state helpers ---------------------------------------


def _new_state():
    """Fresh per-request containers for a ``Plugin.handle`` sequence."""
    return {
        "pending_audio": {},
        "pending_requests": {},
        "pending_dropped": {},
        "cancelled": CancelledTombstones(),
    }


def _messages(blob):
    """Decode the JSON control frames written to an in-memory stream."""
    messages = []
    stream = io.BytesIO(blob)
    while True:
        try:
            kind, payload = read_frame(stream)
        except EOFError:
            return messages
        if kind == KIND_JSON:
            messages.append(decode_json(payload))


def _drain_expecting_error(generator):
    """Drain a generator, returning how it ended and how many frames ran."""
    frames = 0
    try:
        for _frame in generator:
            frames += 1
    except AdapterError as err:
        return "error", frames, f"AdapterError({err.wire_class()})"
    except ValueError:
        return "error", frames, "ValueError"
    except Exception as err:  # noqa: BLE001 - a probe must classify, not escape
        return "other", frames, type(err).__name__
    return "no-error", frames, None


def _invoke(provider, modality):
    """Run the provider's modality operation; return its outcome and value."""
    if modality == "stt":
        generator = provider.transcribe(_PROBE_PCM)
    elif modality == "llm":
        generator = provider.generate(_PROBE_PROMPT)
    elif modality == "tts":
        generator = provider.synthesize(_PROBE_TEXT)
    elif modality == "vad":
        generator = provider.analyze(_PROBE_PCM)
    else:
        return "no-modality", None
    try:
        for _item in generator:
            pass
    except AdapterError as err:
        return "AdapterError", err
    except ValueError as err:
        return "ValueError", err
    except Exception as err:  # noqa: BLE001 - report the type, never its text
        return type(err).__name__, err
    return "no-error", None


# -- probes --------------------------------------------------------------


def _probe_stt_partials(subject, fixtures_dir):
    provider = _provider_for(subject, "stt", fixtures_dir)
    chunks = list(provider.transcribe(_PROBE_PCM))
    final_indices = []
    partials = 0
    well_formed = True
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or "final" not in chunk:
            well_formed = False
            continue
        if chunk.get("final") is True:
            final_indices.append(index)
        elif chunk.get("final") is False:
            partials += 1
        else:
            well_formed = False
    finals = len(final_indices)
    final_last = bool(final_indices) and final_indices[-1] == len(chunks) - 1
    accounted = partials + finals == len(chunks)
    passed = well_formed and accounted and finals == 1 and final_last
    detail = (
        f"chunks={len(chunks)} partials={partials} finals={finals} "
        f"final_last={final_last} well_formed={well_formed}"
    )
    return passed, detail


def _probe_tts_empty_text(subject, fixtures_dir):
    provider = _provider_for(subject, "tts", fixtures_dir)
    observations = []
    passed = True
    for position, text in enumerate(("", "   ")):
        outcome, frames, error_name = _drain_expecting_error(provider.synthesize(text))
        ok = outcome == "error" and frames == 0
        observations.append(
            f"blank#{position}={error_name or outcome} frames={frames}"
        )
        passed = passed and ok
    return passed, "; ".join(observations)


def _probe_tts_no_unrequested_audio(subject, fixtures_dir):
    provider = _provider_for(subject, "tts", fixtures_dir)
    cls = type(provider)
    if not _accepts(_constructor_params(cls), "transport"):
        # Transport-less providers cannot replay an empty body. The
        # deterministic branch below is the documented substitute: the same
        # text yields the same non-zero frame count on both calls.
        first = list(provider.synthesize(_PROBE_TEXT))
        second = list(provider.synthesize(_PROBE_TEXT))
        passed = len(first) == len(second) and len(first) > 0
        return passed, (
            f"branch=deterministic frames={len(first)} repeat={len(second)}"
        )
    empty = _construct_adapter(cls, _empty_body_transport())
    frames = list(empty.synthesize(_PROBE_TEXT))
    return len(frames) == 0, f"branch=empty-body frames={len(frames)}"


def _probe_llm_history(subject, fixtures_dir):
    provider = _provider_for(subject, "llm", fixtures_dir)
    if not _accepts(_callable_params(provider.generate), "history"):
        return False, "branch=structured-history unsupported=True"
    deltas = list(provider.generate(_PROBE_PROMPT, history=list(_PROBE_HISTORY)))
    return len(deltas) >= 1, f"branch=structured-history deltas={len(deltas)}"


def _probe_cancellation(subject, fixtures_dir):
    if subject.primary is not None:
        cls = type(subject.primary)
        if _accepts(_constructor_params(cls), "cancel"):
            token = CancelToken()
            token.cancel()
            provider = _construct_adapter(
                cls, _fixture_transport(fixtures_dir), cancel=token
            )
            outcome, value = _invoke(provider, subject.modality)
            cancelled = (
                outcome == "AdapterError"
                and value.classification is ErrorClass.CANCELLED
            )
            if cancelled:
                return True, "branch=constructor-token classification=cancelled"
            return False, (
                f"branch=constructor-token expected=cancelled observed={outcome}"
            )
    return _probe_plugin_cancellation(subject)


def _probe_plugin_cancellation(subject):
    plugin = subject.plugin
    out = io.BytesIO()
    state = _new_state()
    plugin.handle({"type": "transcribe", "request_id": _CANCEL_ID}, out, **state)
    plugin.handle({"type": "cancel", "request_id": _CANCEL_ID}, out, **state)
    plugin.handle({"type": "audio_end", "request_id": _CANCEL_ID}, out, **state)
    messages = _messages(out.getvalue())
    acks = [
        message
        for message in messages
        if message.get("type") == "cancel_ack"
        and message.get("request_id") == _CANCEL_ID
    ]
    finals = [
        message
        for message in messages
        if message.get("type") == "final" and message.get("request_id") == _CANCEL_ID
    ]
    dones = [
        message
        for message in messages
        if message.get("type") == "done" and message.get("request_id") == _CANCEL_ID
    ]
    passed = len(acks) == 1 and not finals and not dones
    return passed, (
        f"branch=plugin-tombstone ack={len(acks)} finals={len(finals)} "
        f"done={len(dones)}"
    )


def _probe_errors_classified(subject, fixtures_dir):
    if subject.primary is not None and _accepts(
        _constructor_params(type(subject.primary)), "transport"
    ):
        provider = _construct_adapter(type(subject.primary), _catch_all_transport())
        outcome, value = _invoke(provider, subject.modality)
        if outcome == "AdapterError":
            wire = value.wire_class()
            return wire in _WIRE_CLASSES, f"branch=classified-error wire_class={wire}"
        return False, f"branch=classified-error observed={outcome}"
    return _probe_plugin_value_error(subject)


def _probe_plugin_value_error(subject):
    plugin = subject.plugin
    out = io.BytesIO()
    state = _new_state()
    plugin.handle({"type": "transcribe", "request_id": _ERROR_ID}, out, **state)
    # Empty audio makes a bundled fake raise ``ValueError``; the plugin must map
    # that to the ``protocol`` wire class rather than leaking it.
    state["pending_audio"][_ERROR_ID] = b""
    plugin.handle({"type": "audio_end", "request_id": _ERROR_ID}, out, **state)
    errors = [
        message
        for message in _messages(out.getvalue())
        if message.get("type") == "error"
    ]
    wire = errors[0].get("class") if len(errors) == 1 else None
    passed = len(errors) == 1 and wire == "protocol"
    return passed, f"branch=transport-less wire_class={wire}"


def _probe_drop_counters(subject):
    """Drive the host-owned pending-audio buffer on a host-built plugin.

    The counter belongs to ``Plugin`` itself, so it is probed on an instance the
    kit owns: ``Plugin()`` — bundled fakes, no transport, no credential. A
    target ``Plugin`` subclass whose constructor is still exactly
    ``Plugin.__init__`` is rebuilt as a fresh instance of that class, still from
    the bundled fakes, so an under-reporting ``_send`` is exercised. A class
    with a custom constructor is never called (it could dial or demand a
    credential); the plain ``Plugin()`` is used instead. Neither
    ``subject.plugin`` nor any adapter class or transport is read here.

    The four slots are checked to be the exact bundled fakes before a frame is
    driven. When the counter cannot be read, the failure detail carries counts
    and wire classes only — never frame text, a provider message or a
    credential.
    """
    plugin_cls = subject.plugin_cls
    if plugin_cls.__init__ is Plugin.__init__:
        plugin = plugin_cls()
    else:
        plugin = Plugin()
    if not all(
        type(getattr(plugin, modality, None)) in _STOCK_FAKES
        for modality in _MODALITIES
    ):
        return False, (
            f"frames=0 done=0 errors=0 wire_classes=none dropped=None "
            f"expected={_EXPECTED_DROPS} cause=non-fake-slots"
        )
    # Every probe gets a fresh state mapping and local buffers, so request
    # state does not leak between probes.
    out = io.BytesIO()
    state = _new_state()
    plugin.handle({"type": "transcribe", "request_id": _DROP_ID}, out, **state)
    frame = b"\x00\x00" * 160
    for _ in range(_AUDIO_PROBE_FRAMES):
        plugin._buffer_audio(
            _DROP_ID,
            frame,
            state["pending_audio"],
            state["pending_requests"],
            state["pending_dropped"],
            state["cancelled"],
        )
    plugin.handle({"type": "audio_end", "request_id": _DROP_ID}, out, **state)
    messages = _messages(out.getvalue())
    dones = [
        message
        for message in messages
        if message.get("type") == "done" and message.get("request_id") == _DROP_ID
    ]
    errors = [
        message
        for message in messages
        if message.get("type") == "error" and message.get("request_id") == _DROP_ID
    ]
    dropped = dones[0].get("dropped_frames") if len(dones) == 1 else None
    passed = len(dones) == 1 and dropped == _EXPECTED_DROPS and not errors
    if passed:
        return True, (
            f"frames={_AUDIO_PROBE_FRAMES} done=1 errors=0 "
            f"dropped={dropped} expected={_EXPECTED_DROPS}"
        )
    wire_classes = sorted(
        {str(message.get("class") or "unclassified") for message in errors}
    )
    if not dones:
        cause = "error-frame-no-done" if errors else "no-done-frame"
    elif len(dones) > 1:
        cause = "multiple-done-frames"
    elif errors:
        cause = "done-with-error-frames"
    else:
        cause = "under-reported-drops"
    return False, (
        f"frames={_AUDIO_PROBE_FRAMES} done={len(dones)} errors={len(errors)} "
        f"wire_classes={','.join(wire_classes) if wire_classes else 'none'} "
        f"dropped={dropped} expected={_EXPECTED_DROPS} cause={cause}"
    )


# -- entry point ---------------------------------------------------------


def _safe(invariant, probe):
    """Run one probe, converting anything it raises into a failed result."""
    try:
        passed, detail = probe()
    except AdapterConfigError:
        # Configuration defects are loud: a bad fixture directory or an
        # unusable target is the caller's error, not an adapter verdict.
        raise
    except AdapterError as err:
        passed = False
        detail = f"probe raised AdapterError wire_class={err.wire_class()}"
    except Exception as err:  # noqa: BLE001 - no probe failure may escape run()
        passed = False
        detail = f"probe raised {type(err).__name__}"
    return ConformanceResult(invariant, bool(passed), redact(detail))


def run(target, *, fixtures_dir):
    """Check ``target`` against all seven invariants.

    Returns one :class:`ConformanceResult` per entry in
    :data:`CONFORMANCE_INVARIANTS`, in order. Probe failures are reported as
    results. The target is normalised first: a class, a transport-less
    provider, or a provider not wired to a ``FixtureTransport`` raises
    :class:`ConformanceTargetError`. The fixture directory is then opened
    through the kit's replay-only transport, so a missing directory — like an
    unrecognised modality — raises
    :class:`~parlevox_sdk.adapters.base.AdapterConfigError`. Every refusal is
    raised before any probe runs.
    """
    subject = _normalise(target, fixtures_dir)
    _fixture_transport(fixtures_dir)
    probes = (
        (
            "stt-partials-then-one-final",
            lambda: _probe_stt_partials(subject, fixtures_dir),
        ),
        (
            "tts-empty-text-refused",
            lambda: _probe_tts_empty_text(subject, fixtures_dir),
        ),
        (
            "tts-no-unrequested-audio",
            lambda: _probe_tts_no_unrequested_audio(subject, fixtures_dir),
        ),
        (
            "llm-structured-history-accepted",
            lambda: _probe_llm_history(subject, fixtures_dir),
        ),
        (
            "cancellation-propagates",
            lambda: _probe_cancellation(subject, fixtures_dir),
        ),
        (
            "errors-classified",
            lambda: _probe_errors_classified(subject, fixtures_dir),
        ),
        (
            "drop-counters-honest",
            lambda: _probe_drop_counters(subject),
        ),
    )
    return [_safe(invariant, probe) for invariant, probe in probes]
