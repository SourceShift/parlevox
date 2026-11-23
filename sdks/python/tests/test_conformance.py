"""The sidecar adapter conformance kit: red lines, baseline, offline proof.

The kit is exercised three ways:

* the bundled fakes are the always-on offline baseline, so a run is never
  silent;
* seven deliberately broken providers — one per invariant — must each fail
  exactly their own invariant and pass the other six;
* when ``PARLEVOX_CONFORMANCE_ADAPTERS`` names registered adapters, each one is
  replayed against its vendor cassettes and must pass all seven. Unset, that
  parametrisation skips with an explicit message and nothing else changes.

The target forms are pinned too: only already-constructed instances are
accepted, and a raising socket sentinel counts every dial attempt across an
accepted form and a refused form, so the offline guarantee is proven rather than
assumed. No test opens a socket, writes a file, or reads repository state.
"""

import importlib.util
import http.client
import os
import pathlib
import re
import socket
import subprocess
import sys

import pytest

from parlevox_sdk.adapters import AdapterConfigError, FixtureTransport, OpenAISTT
from parlevox_sdk.conformance import (
    CONFORMANCE_INVARIANTS,
    PLACEHOLDER_KEY,
    ConformanceResult,
    ConformanceTargetError,
    run,
)
from parlevox_sdk.fakes import FakeLlm, FakeStt, FakeTts
from parlevox_sdk.plugin import Plugin

TESTS_DIR = pathlib.Path(__file__).resolve().parent
SDK_ROOT = TESTS_DIR.parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
BROKEN_DIR = FIXTURES_DIR / "broken"

#: Vendor fixture directory for each registered adapter, keyed ``modality:name``.
ADAPTER_FIXTURES = {
    "stt:openai": "openai",
    "stt:deepgram": "deepgram",
    "stt:assemblyai": "assemblyai",
    "llm:openai": "openai",
    "tts:openai": "openai",
    "tts:elevenlabs": "elevenlabs",
    "tts:cartesia": "cartesia",
}

#: One deliberately broken provider per invariant: (invariant, module, directory).
_BROKEN_SPECS = (
    ("stt-partials-then-one-final", "stt_partials_then_one_final.py", "openai"),
    ("tts-empty-text-refused", "tts_empty_text_refused.py", "openai"),
    ("tts-no-unrequested-audio", "tts_no_unrequested_audio.py", "elevenlabs"),
    ("llm-structured-history-accepted", "llm_structured_history_accepted.py", "openai"),
    ("cancellation-propagates", "cancellation_propagates.py", "openai"),
    ("errors-classified", "errors_classified.py", "deepgram"),
    ("drop-counters-honest", "drop_counters_honest.py", "openai"),
)

#: Credential shapes that must never appear in a result's detail or rendering.
_CREDENTIAL_SHAPES = (
    r"sk-[A-Za-z0-9_-]{24,}",
    r"ghp_[A-Za-z0-9]{30,}",
    r"github_pat_[A-Za-z0-9_]{30,}",
    r"AKIA[0-9A-Z]{16}",
    r"xox[bpa]-[A-Za-z0-9-]{10,}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY",
)


class _DialingAdapter:
    """A transport-less provider that would dial if the kit ever ran it."""

    manifest = {"modality": "stt"}

    def transcribe(self, _pcm, **_kwargs):
        socket.create_connection(("127.0.0.1", 9), timeout=0.1)
        yield {"text": "", "final": True}


class _DialingPlugin(Plugin):
    """A plugin whose STT slot is a transport-less dialer."""

    def __init__(self):
        super().__init__()
        self.stt = _DialingAdapter()


class _ErroringPlugin(Plugin):
    """A plugin whose ``audio_end`` always reports a protocol error."""

    def _finish_audio_request(
        self,
        request_id,
        out,
        pending_audio,
        pending_requests,
        pending_dropped,
        cancelled,
    ):
        self._send_error(out, request_id, "protocol", "synthetic failure")
        return True


class _SuppressingDone(Plugin):
    """A plugin that never emits a terminal ``done`` message."""

    @staticmethod
    def _send(out, message):
        if isinstance(message, dict) and message.get("type") == "done":
            return
        Plugin._send(out, message)


def _load_broken(filename):
    """Import a broken-provider module by path, without making it a package."""
    path = BROKEN_DIR / filename
    name = "conformance_broken_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_transport(fixtures_dir):
    """A replay-only transport over ``fixtures_dir``, as the kit requires."""
    return FixtureTransport.from_directory(
        fixtures_dir,
        provider="conformance",
        max_attempts=1,
        sleeper=lambda _seconds: None,
    )


def _construct_broken(cls, fixtures_dir):
    """Instantiate a broken provider in one of the kit's two accepted forms.

    The placeholder credential mirrors how the kit builds probe providers. An
    adapter resolves its key only at first request, so constructing the target
    stays credential-free and offline.
    """
    if issubclass(cls, Plugin):
        return cls()
    env_var = getattr(cls, "ENV_VAR", "")
    environ = {env_var: PLACEHOLDER_KEY} if env_var else {}
    return cls(transport=_fixture_transport(fixtures_dir), environ=environ)


def _broken_cases():
    cases = []
    targets = []
    for invariant, filename, directory in _BROKEN_SPECS:
        module = _load_broken(filename)
        assert module.PROVIDER is not None, filename
        assert (FIXTURES_DIR / module.FIXTURES).is_dir(), filename
        fixtures_dir = FIXTURES_DIR / directory
        # The kit refuses classes: every broken target must be an already
        # constructed instance over a fixture transport (or a Plugin instance).
        provider = _construct_broken(module.PROVIDER, fixtures_dir)
        assert not isinstance(provider, type), filename
        targets.append((provider, fixtures_dir))
        cases.append(pytest.param(invariant, provider, fixtures_dir, id=invariant))
    return cases, targets


BROKEN_CASES, BROKEN_TARGETS = _broken_cases()

_BASELINE_CASES = (
    pytest.param(FakeStt(), FIXTURES_DIR / "openai", id="fake-stt"),
    pytest.param(FakeLlm(), FIXTURES_DIR / "openai", id="fake-llm"),
    pytest.param(FakeTts(), FIXTURES_DIR / "openai", id="fake-tts"),
)
ALL_TARGETS = list(_BASELINE_CASES) + [
    pytest.param(provider, fixtures_dir, id=invariant)
    for (invariant, _filename, _directory), (provider, fixtures_dir) in zip(
        _BROKEN_SPECS, BROKEN_TARGETS
    )
]

_SELECTED = [
    token.strip()
    for token in os.environ.get("PARLEVOX_CONFORMANCE_ADAPTERS", "").split(",")
    if token.strip()
]


@pytest.mark.parametrize("invariant, provider, fixtures_dir", BROKEN_CASES)
def test_broken_provider_fails_exactly_one_invariant(invariant, provider, fixtures_dir):
    results = run(provider, fixtures_dir=fixtures_dir)
    assert [result.invariant for result in results] == list(CONFORMANCE_INVARIANTS)
    failed = [result.invariant for result in results if not result.passed]
    assert failed == [invariant], [
        (result.invariant, result.passed, result.detail) for result in results
    ]
    for result in results:
        if result.invariant != invariant:
            assert result.passed, result
    # Visible with ``-s``: one line per provider, plus every observed detail.
    print(f"broken {invariant}: failed=[{', '.join(failed)}]")
    print(
        "conformance:",
        [(result.invariant, result.passed, result.detail) for result in results],
    )


def test_fake_stt_passes_all_seven():
    results = run(FakeStt(), fixtures_dir=FIXTURES_DIR / "openai")
    assert [result.invariant for result in results] == list(CONFORMANCE_INVARIANTS)
    assert all(result.passed for result in results), [
        (result.invariant, result.detail) for result in results
    ]


@pytest.mark.parametrize("provider, fixtures_dir", ALL_TARGETS)
def test_run_returns_one_result_per_invariant(provider, fixtures_dir):
    results = run(provider, fixtures_dir=fixtures_dir)
    invariants = [result.invariant for result in results]
    assert invariants == list(CONFORMANCE_INVARIANTS)
    assert len(set(invariants)) == len(CONFORMANCE_INVARIANTS)
    assert all(isinstance(result, ConformanceResult) for result in results)


def test_run_is_deterministic():
    first = run(FakeStt(), fixtures_dir=FIXTURES_DIR / "openai")
    second = run(FakeStt(), fixtures_dir=FIXTURES_DIR / "openai")
    assert [(result.invariant, result.passed) for result in first] == [
        (result.invariant, result.passed) for result in second
    ]


@pytest.mark.parametrize("provider, fixtures_dir", ALL_TARGETS)
def test_details_never_carry_credentials(provider, fixtures_dir):
    results = run(provider, fixtures_dir=fixtures_dir)
    for result in results:
        rendered = "\n".join((result.detail, str(result), repr(result)))
        assert PLACEHOLDER_KEY not in rendered
        for shape in _CREDENTIAL_SHAPES:
            assert re.search(shape, rendered) is None, (
                result.invariant,
                result.detail,
            )


def test_conformance_imports_without_pytest():
    code = "\n".join(
        (
            "import builtins",
            "real_import = builtins.__import__",
            "def blocked(name, *args, **kwargs):",
            "    if name == 'pytest' or name.startswith('pytest.'):",
            "        raise AssertionError('conformance imported pytest')",
            "    return real_import(name, *args, **kwargs)",
            "builtins.__import__ = blocked",
            "import parlevox_sdk.conformance as conformance",
            "assert len(conformance.CONFORMANCE_INVARIANTS) == 7",
            "assert len(set(conformance.CONFORMANCE_INVARIANTS)) == 7",
        )
    )
    environ = dict(os.environ)
    environ["PYTHONPATH"] = str(SDK_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=environ,
    )
    assert completed.returncode == 0, completed.stderr


def test_run_refuses_non_fixture_transport_without_dialing(monkeypatch):
    import http.client
    import socket

    def refuse(*_args, **_kwargs):
        raise AssertionError("network dial attempted")

    monkeypatch.setattr(http.client, "HTTPConnection", refuse)
    monkeypatch.setattr(http.client, "HTTPSConnection", refuse)
    monkeypatch.setattr(socket, "socket", refuse)

    # No PARLEVOX_FIXTURE_DIR: this adapter is wired to a real HTTP transport,
    # so the kit must refuse it before any probe can reach the wire.
    adapter = OpenAISTT(
        base_url="http://127.0.0.1:9",
        environ={"PARLEVOX_OPENAI_API_KEY": PLACEHOLDER_KEY},
    )
    with pytest.raises(AdapterConfigError):
        run(adapter, fixtures_dir=FIXTURES_DIR / "openai")

    baseline = run(FakeStt(), fixtures_dir=FIXTURES_DIR / "openai")
    assert all(result.passed for result in baseline)

    for provider, fixtures_dir in BROKEN_TARGETS:
        results = run(provider, fixtures_dir=fixtures_dir)
        assert [result.invariant for result in results] == list(
            CONFORMANCE_INVARIANTS
        )


def test_socket_sentinel_zero_dials_across_target_forms(monkeypatch):
    """Every accepted form runs offline; every refused form is refused offline.

    Each socket entry point raises and increments ``dials``. A single dial in
    either phase fails the test, so the offline guarantee covers the refusal
    path and not only the happy path.
    """
    dials = []

    def refuse(*_args, **_kwargs):
        dials.append(1)
        raise AssertionError("network dial attempted")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(http.client, "HTTPConnection", refuse)
    monkeypatch.setattr(http.client, "HTTPSConnection", refuse)

    fixtures_dir = FIXTURES_DIR / "openai"
    fixture_stt = OpenAISTT(
        transport=_fixture_transport(fixtures_dir),
        environ={"PARLEVOX_OPENAI_API_KEY": PLACEHOLDER_KEY},
    )
    accepted = {
        "fake-adapter-instance": FakeStt(),
        "fixture-adapter-instance": fixture_stt,
        "plugin-with-fixture-adapter": Plugin(stt=fixture_stt),
        "plugin-from-env": Plugin.from_env(
            environ={
                "PARLEVOX_ADAPTERS": "stt:openai",
                "PARLEVOX_FIXTURE_DIR": str(fixtures_dir),
                "PARLEVOX_OPENAI_API_KEY": PLACEHOLDER_KEY,
            }
        ),
    }
    # The drop probe is host-built for every target, so it must count exactly
    # three drops without touching the target's adapter.
    for label, target in accepted.items():
        results = run(target, fixtures_dir=fixtures_dir)
        assert [result.invariant for result in results] == list(
            CONFORMANCE_INVARIANTS
        ), label
        drop = next(
            result
            for result in results
            if result.invariant == "drop-counters-honest"
        )
        assert drop.passed, (label, drop.detail)
        assert "dropped=3 " in drop.detail, (label, drop.detail)

    http_backed = OpenAISTT(
        base_url="http://127.0.0.1:9",
        environ={"PARLEVOX_OPENAI_API_KEY": PLACEHOLDER_KEY},
    )
    refused = {
        "adapter-class": OpenAISTT,
        "plugin-class": Plugin,
        "http-transport-instance": http_backed,
        "dialing-adapter-class": _DialingAdapter,
        "dialing-adapter-instance": _DialingAdapter(),
        "dialing-plugin-instance": _DialingPlugin(),
    }
    for label, target in refused.items():
        with pytest.raises(ConformanceTargetError):
            run(target, fixtures_dir=fixtures_dir)

    assert not dials, f"dial attempts: {len(dials)}"
    print(
        "conformance sentinel:",
        f"dials={len(dials)} accepted={len(accepted)} refused={len(refused)}",
    )
    print("conformance sentinel accepted:", sorted(accepted))
    print("conformance sentinel refused:", sorted(refused))


_REFUSED_FORMS = (
    pytest.param(OpenAISTT, id="class"),
    pytest.param(_DialingAdapter(), id="transport-less-instance"),
    pytest.param(
        OpenAISTT(
            base_url="http://127.0.0.1:9",
            environ={"PARLEVOX_OPENAI_API_KEY": PLACEHOLDER_KEY},
        ),
        id="http-transport-instance",
    ),
)


@pytest.mark.parametrize("target", _REFUSED_FORMS)
def test_class_target_refused_names_accepted_forms(target):
    """Every refused form is refused with a message naming both accepted forms."""
    with pytest.raises(ConformanceTargetError) as excinfo:
        run(target, fixtures_dir=FIXTURES_DIR / "openai")
    message = str(excinfo.value)
    label = target.__name__ if isinstance(target, type) else type(target).__name__
    print(f"conformance refusal {label}: {message}")
    assert "Plugin instance" in message, (label, message)
    assert "adapter instance" in message, (label, message)
    assert "FixtureTransport" in message, (label, message)
    assert PLACEHOLDER_KEY not in message, (label, message)
    for shape in _CREDENTIAL_SHAPES:
        assert re.search(shape, message) is None, (label, message)


def test_missing_fixture_directory_refused_before_probe(monkeypatch):
    """A missing fixture directory is refused before probe 1 is even called."""
    import parlevox_sdk.conformance as conformance

    def explode(*_args, **_kwargs):
        raise AssertionError("a probe ran before the fixture-directory check")

    monkeypatch.setattr(conformance, "_probe_stt_partials", explode)

    with pytest.raises(AdapterConfigError) as excinfo:
        run(FakeStt(), fixtures_dir=pathlib.Path("/nonexistent/conformance-fixtures"))
    message = str(excinfo.value)
    print("conformance missing-fixtures:", message)
    assert "fixture directory not found: conformance-fixtures" in message
    assert "/nonexistent" not in message, message
    assert PLACEHOLDER_KEY not in message
    for shape in _CREDENTIAL_SHAPES:
        assert re.search(shape, message) is None, message


def test_drop_probe_names_error_frame_cause():
    """An error frame with no ``done`` is named by counts and wire class."""
    results = run(_ErroringPlugin(), fixtures_dir=FIXTURES_DIR / "openai")
    drop = next(
        result for result in results if result.invariant == "drop-counters-honest"
    )
    print("conformance drop-cause:", drop.detail)
    assert drop.passed is False
    assert "frames=67 " in drop.detail
    assert "done=0 " in drop.detail
    assert "errors=1 " in drop.detail
    assert "wire_classes=protocol " in drop.detail
    assert "dropped=None " in drop.detail
    assert "expected=3 " in drop.detail
    assert "cause=error-frame-no-done" in drop.detail
    assert PLACEHOLDER_KEY not in drop.detail
    for shape in _CREDENTIAL_SHAPES:
        assert re.search(shape, drop.detail) is None, drop.detail


def test_drop_probe_names_missing_done_cause():
    """A suppressed ``done`` is named as ``done=0`` with the cause."""
    results = run(_SuppressingDone(), fixtures_dir=FIXTURES_DIR / "openai")
    drop = next(
        result for result in results if result.invariant == "drop-counters-honest"
    )
    print("conformance drop-missing-done:", drop.detail)
    assert drop.passed is False
    assert "done=0 " in drop.detail
    assert "errors=0 " in drop.detail
    assert "wire_classes=none " in drop.detail
    assert "dropped=None " in drop.detail
    assert "expected=3 " in drop.detail
    assert "cause=no-done-frame" in drop.detail
    assert PLACEHOLDER_KEY not in drop.detail
    for shape in _CREDENTIAL_SHAPES:
        assert re.search(shape, drop.detail) is None, drop.detail


def test_drop_probe_does_not_construct_custom_plugin_constructor():
    """A plugin class with its own constructor is never called by the probe."""

    class NeedsArgument(Plugin):
        def __init__(self, required):
            super().__init__()
            self.required = required

    results = run(NeedsArgument(required=True), fixtures_dir=FIXTURES_DIR / "openai")
    drop = next(
        result for result in results if result.invariant == "drop-counters-honest"
    )
    print("conformance drop-custom-constructor:", drop.detail)
    assert drop.passed is True, drop.detail
    assert "done=1 " in drop.detail
    assert "errors=0 " in drop.detail
    assert "dropped=3 " in drop.detail


def test_details_and_logs_never_carry_credentials(caplog):
    """Neither a detail nor any captured log carries key material."""
    import logging

    targets = [(FakeStt(), FIXTURES_DIR / "openai")] + list(BROKEN_TARGETS)
    with caplog.at_level(logging.DEBUG):
        for provider, fixtures_dir in targets:
            results = run(provider, fixtures_dir=fixtures_dir)
            for result in results:
                rendered = "\n".join(
                    (result.invariant, result.detail, str(result), repr(result))
                )
                assert PLACEHOLDER_KEY not in rendered, (
                    result.invariant,
                    rendered,
                )
                for shape in _CREDENTIAL_SHAPES:
                    assert re.search(shape, rendered) is None, (
                        result.invariant,
                        result.detail,
                    )
    logged = caplog.text
    assert PLACEHOLDER_KEY not in logged
    for shape in _CREDENTIAL_SHAPES:
        assert re.search(shape, logged) is None, logged


@pytest.mark.parametrize("token", _SELECTED or [None])
def test_registered_adapter_conforms(token):
    if not _SELECTED:
        pytest.skip(
            "PARLEVOX_CONFORMANCE_ADAPTERS is unset; set it to a comma-separated "
            "modality:name list to replay registered adapters. The fake "
            "baseline and the broken-provider matrix always run."
        )
    if token not in ADAPTER_FIXTURES:
        pytest.fail(
            f"unknown adapter {token!r}; known adapters: {sorted(ADAPTER_FIXTURES)}"
        )
    fixtures_dir = FIXTURES_DIR / ADAPTER_FIXTURES[token]
    plugin = Plugin.from_env(
        environ={
            "PARLEVOX_ADAPTERS": token,
            "PARLEVOX_FIXTURE_DIR": str(fixtures_dir),
        }
    )
    results = run(plugin, fixtures_dir=fixtures_dir)
    assert [result.invariant for result in results] == list(CONFORMANCE_INVARIANTS)
    # Visible with ``-s``: the seven invariant tuples the kit reports.
    print(
        "conformance:",
        [(result.invariant, result.passed, result.detail) for result in results],
    )
    failed = [result for result in results if not result.passed]
    assert not failed, [(result.invariant, result.detail) for result in failed]
    drop = next(
        result for result in results if result.invariant == "drop-counters-honest"
    )
    assert "dropped=3 " in drop.detail, drop.detail
    assert "done=1 " in drop.detail, drop.detail
    assert "errors=0 " in drop.detail, drop.detail
