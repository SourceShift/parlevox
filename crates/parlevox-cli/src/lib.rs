//! `parlevox-cli` — one-turn demo over either provider backend.
//!
//! The argument parser and demo runners live here so integration tests can
//! drive the same code path the binary does. `main.rs` is a thin delegate.
//!
//! `demo --fake` uses only in-process fakes: no process, no Python import, no
//! network. `demo --sidecar` runs the same turn with STT, LLM and TTS served by
//! the optional Python plugin over the framed protocol.

use std::path::PathBuf;
use std::sync::Arc;

use parlevox_core::ParlevoxError;

/// Opt-in OTLP/HTTP trace export, enabled by the `otlp` cargo feature.
#[cfg(feature = "otlp")]
pub mod export;
use parlevox_providers::{
    default_layers, FakeVad, LlmProvider, ProviderDescriptor,
    ProviderRegistry as DescriptorRegistry, SttProvider, TtsProvider,
};
use parlevox_runtime::{
    demo_utterance, ProviderNames, ProviderRegistry, Providers, Recorder, RecorderConfig,
    RecorderStats, Session, SessionConfig, SessionEvent, TurnReport,
};
use parlevox_sidecar::{
    resolve_sdk_path, PoolConfig, SdkSearch, SidecarClient, SidecarConfig, SidecarLlm, SidecarPool,
    SidecarStt, SidecarTts,
};

/// Build a session whose STT, LLM and TTS are served by `client` over the
/// framed protocol; VAD stays in process because it is a deterministic fake.
///
/// This is the single construction path shared by the differential parity test
/// and the pool reuse test, so a change to provider wiring cannot make those
/// tests disagree about what a session is.
///
/// # Errors
///
/// Returns a protocol-class error when a server-backed provider's manifest is
/// invalid or the requested names cannot be resolved, and any error from
/// session configuration.
pub fn sidecar_session(client: Arc<SidecarClient>) -> Result<Session, ParlevoxError> {
    let (session, _names) = build_sidecar_session(client)?;
    Ok(session)
}

/// Build the sidecar-backed session and return the provider names beside it.
///
/// The names are what the demo prints, so the pooled path does not have to
/// re-derive them from a second construction path.
///
/// # Errors
///
/// Same as [`sidecar_session`]: an invalid manifest, unresolvable names, or a
/// rejected session configuration.
fn build_sidecar_session(
    client: Arc<SidecarClient>,
) -> Result<(Session, ProviderNames), ParlevoxError> {
    let stt = Arc::new(SidecarStt::new(Arc::clone(&client)));
    let llm = Arc::new(SidecarLlm::new(Arc::clone(&client)));
    let tts = Arc::new(SidecarTts::new(Arc::clone(&client)));
    let mut registry = ProviderRegistry::new();
    registry.register_stt(stt.clone())?;
    registry.register_llm(llm.clone())?;
    registry.register_tts(tts.clone())?;
    registry.register_vad(Arc::new(FakeVad::new()))?;

    let names = ProviderNames {
        stt: stt.manifest().name.clone(),
        llm: llm.manifest().name.clone(),
        tts: tts.manifest().name.clone(),
        vad: "fake-vad".into(),
    };
    let providers = registry.resolve(&names)?;
    let session = build_session(SessionConfig::default(), providers)?;
    Ok((session, names))
}

/// Build a session, routing it through the installed OTLP tracer and meter
/// when the `otlp` feature is on.
///
/// Without the feature, or before `export::install` has run, this is exactly
/// [`Session::new`] and the session owns a no-op global tracer. With an
/// installed exporter the session records its turn spans and the six turn
/// metrics through the process-global providers; if no meter is available it
/// falls back to the tracer-only session.
///
/// # Errors
///
/// Returns the same protocol-class configuration errors as [`Session::new`].
fn build_session(config: SessionConfig, providers: Providers) -> Result<Session, ParlevoxError> {
    #[cfg(feature = "otlp")]
    {
        if let Some(tracer) = export::tracer() {
            let session = Session::with_tracer(config, providers, tracer)?;
            return Ok(match export::meter() {
                Some(meter) => session.with_meter(meter),
                None => session,
            });
        }
    }
    Session::new(config, providers)
}

/// Install the OTLP exporter from the environment when the `otlp` feature is
/// on, printing exactly one status line.
fn start_trace_export() {
    #[cfg(feature = "otlp")]
    match export::TraceExportConfig::from_env() {
        Some(config) => match export::install(&config) {
            Ok(()) => eprintln!(
                "otlp traces -> {} service={}",
                config.endpoint, config.service_name
            ),
            Err(err) => eprintln!("otlp exporter failed: {err}"),
        },
        None => eprintln!("otlp exporter disabled: OTEL_EXPORTER_OTLP_ENDPOINT is not set"),
    }
}

/// Flush and shut the installed exporter down. A no-op without the `otlp`
/// feature, and never an error: export failure must not fail a finished turn.
fn finish_trace_export() {
    #[cfg(feature = "otlp")]
    export::flush_and_shutdown();
}

/// Canonical, timestamp-free projection of an event trace.
///
/// Audio timestamps are session-epoch-relative, so they are normalized away;
/// kind, turn, text, PCM samples, sample rate, channels and every drop counter
/// are preserved. Two runs are interchangeable exactly when their projections
/// are equal.
#[must_use]
pub fn project_events(events: &[SessionEvent]) -> Vec<String> {
    events
        .iter()
        .map(|event| match event {
            SessionEvent::TurnStarted { turn } => format!("TurnStarted turn={turn}"),
            SessionEvent::TranscriptFinal { turn, text } => {
                format!("TranscriptFinal turn={turn} text={text:?}")
            }
            SessionEvent::ReplyDelta { turn, text } => {
                format!("ReplyDelta turn={turn} text={text:?}")
            }
            SessionEvent::AudioOut { turn, frame } => format!(
                "AudioOut turn={turn} rate={} channels={} pcm={:?}",
                frame.sample_rate, frame.channels, frame.pcm
            ),
            SessionEvent::TurnDone {
                turn,
                audio_frames,
                dropped_frames,
                dropped_ms,
                dropped_events,
                sidecar_dropped_frames,
                interrupted,
            } => format!(
                "TurnDone turn={turn} audio_frames={audio_frames} \
                 dropped_frames={dropped_frames} dropped_ms={dropped_ms} \
                 dropped_events={dropped_events} \
                 sidecar_dropped_frames={sidecar_dropped_frames} \
                 interrupted={interrupted}"
            ),
            SessionEvent::Failed { class, message } => {
                format!("Failed class={class:?} message={message:?}")
            }
        })
        .collect()
}

/// Which provider backend a `demo` invocation drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DemoBackend {
    Fake,
    Sidecar,
}

/// A `--record` target and its transcript policy.
#[derive(Debug, Clone)]
struct RecordOptions {
    path: PathBuf,
    record_transcripts: bool,
}

/// Parsed `demo` arguments.
#[derive(Debug)]
struct DemoArgs {
    backend: DemoBackend,
    pool: bool,
    record: Option<RecordOptions>,
}

/// Parse `demo` arguments after the `demo` command word.
///
/// The accepted grammar is `--fake|--sidecar [--record <path>]
/// [--record-transcripts] [--pool]`, with `--fake --pool` rejected and
/// `--record-transcripts` required to accompany `--record`.
fn parse_demo(rest: &[String]) -> Result<DemoArgs, ParlevoxError> {
    let mut backend = None;
    let mut record: Option<RecordOptions> = None;
    let mut record_transcripts = false;
    let mut pool = false;
    let mut index = 0;
    while index < rest.len() {
        match rest[index].as_str() {
            flag @ ("--fake" | "--sidecar") => {
                if backend.is_some() {
                    return Err(ParlevoxError::Protocol(format!(
                        "demo backend given more than once: {flag}"
                    )));
                }
                backend = Some(if flag == "--fake" {
                    DemoBackend::Fake
                } else {
                    DemoBackend::Sidecar
                });
            }
            "--record" => {
                index += 1;
                let Some(path) = rest.get(index) else {
                    return Err(ParlevoxError::Protocol(
                        "--record requires a file path: --record <path>".into(),
                    ));
                };
                record = Some(RecordOptions {
                    path: PathBuf::from(path),
                    record_transcripts: false,
                });
            }
            "--record-transcripts" => record_transcripts = true,
            "--pool" => pool = true,
            other => {
                return Err(ParlevoxError::Protocol(format!(
                    "unrecognised demo argument: {other}"
                )));
            }
        }
        index += 1;
    }

    let Some(backend) = backend else {
        return Err(ParlevoxError::Protocol(
            "demo requires --fake or --sidecar".into(),
        ));
    };
    if record_transcripts && record.is_none() {
        return Err(ParlevoxError::Protocol(
            "--record-transcripts requires --record <path>".into(),
        ));
    }
    if record_transcripts {
        if let Some(options) = record.as_mut() {
            options.record_transcripts = true;
        }
    }
    if pool && backend == DemoBackend::Fake {
        return Err(ParlevoxError::Protocol(
            "demo --fake --pool is rejected: --pool requires --sidecar \
             (the fakes spawn no plugin process)"
                .into(),
        ));
    }
    Ok(DemoArgs {
        backend,
        pool,
        record,
    })
}

/// Parse process arguments and run the matching demo.
///
/// # Errors
///
/// Returns a protocol-class error for unrecognised arguments and propagates
/// whatever the selected demo returns (provider, transport, or protocol
/// errors).
pub async fn run_args() -> Result<(), ParlevoxError> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [flag] if flag == "--help" || flag == "-h" => {
            usage();
            Ok(())
        }
        [command, rest @ ..] if command == "demo" => {
            let demo = parse_demo(rest).inspect_err(|_| usage())?;
            run_demo(demo).await
        }
        [command, rest @ ..] if command == "providers" => run_providers(rest),
        _ => {
            usage();
            Err(ParlevoxError::Protocol(format!(
                "unrecognised arguments: {}",
                args.join(" ")
            )))
        }
    }
}

async fn run_demo(args: DemoArgs) -> Result<(), ParlevoxError> {
    if args.pool {
        return demo_sidecar_pooled_with(args.record).await;
    }
    match args.backend {
        DemoBackend::Fake => demo_fake_with(args.record).await,
        DemoBackend::Sidecar => demo_sidecar_with(args.record).await,
    }
}

fn run_providers(rest: &[String]) -> Result<(), ParlevoxError> {
    match rest {
        [subcommand] if subcommand == "list" => providers_list(),
        [subcommand, name] if subcommand == "describe" => providers_describe(name),
        [flag] if flag == "--help" || flag == "-h" => {
            providers_usage();
            Ok(())
        }
        _ => {
            providers_usage();
            Err(ParlevoxError::Protocol(format!(
                "unrecognised providers arguments: {}",
                rest.join(" ")
            )))
        }
    }
}

pub fn usage() {
    eprintln!("usage: parlevox demo --fake");
    eprintln!("       parlevox demo --sidecar");
    eprintln!("       parlevox demo --sidecar --pool");
    eprintln!("       parlevox demo --fake|--sidecar --record <path> [--record-transcripts]");
    eprintln!("note: demo --fake --pool is rejected; --pool requires --sidecar");
}

/// Attach a recorder to `session` when a `--record` target was given.
///
/// The recorder is created before the turn, so an unwritable path fails the
/// command rather than being discovered mid-turn.
fn attach_recorder(
    session: Session,
    record: Option<&RecordOptions>,
) -> Result<Session, ParlevoxError> {
    let Some(record) = record else {
        return Ok(session);
    };
    let config = RecorderConfig {
        record_transcripts: record.record_transcripts,
        ..RecorderConfig::default()
    };
    let recorder = Recorder::to_path(&record.path, config)?;
    Ok(session.with_recorder(recorder))
}

/// Print recorder counters to stderr, only when a recorder is installed.
fn print_recorder_stats(stats: Option<RecorderStats>) {
    if let Some(stats) = stats {
        eprintln!(
            "recorder records={} bytes={} dropped_cap={} dropped_write={} redacted={} \
             transcripts_suppressed={}",
            stats.records_written,
            stats.bytes_written,
            stats.dropped_cap,
            stats.dropped_write,
            stats.redacted,
            stats.transcripts_suppressed
        );
    }
}

/// Print usage for the `providers` subcommands.
pub fn providers_usage() {
    println!("usage: parlevox providers list");
    println!("       parlevox providers describe <name>");
    println!(
        "note: descriptors come from config/providers, PARLEVOX_PROVIDER_DIRS \
         and $PARLEVOX_HOME/providers"
    );
}

fn providers_list() -> Result<(), ParlevoxError> {
    let registry = DescriptorRegistry::load(&default_layers())?;
    print!("{}", render_list(&registry));
    Ok(())
}

fn providers_describe(name: &str) -> Result<(), ParlevoxError> {
    let registry = DescriptorRegistry::load(&default_layers())?;
    let entry = registry.find(name)?;
    let api_key_present = entry
        .descriptor
        .api_key_env
        .as_deref()
        .is_some_and(|variable| std::env::var_os(variable).is_some());
    print!("{}", render_describe(&entry.descriptor, api_key_present));
    Ok(())
}

/// Render one tab-separated row per descriptor: name, modality, tier, adapter
/// kind and the file it was loaded from, sorted by modality then name.
#[must_use]
pub fn render_list(registry: &DescriptorRegistry) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    for entry in registry.all() {
        let descriptor = &entry.descriptor;
        let _ = writeln!(
            out,
            "{}\t{}\tdescriptor\t{}\t{}",
            descriptor.name,
            descriptor.modality.as_str(),
            adapter_kind(descriptor),
            entry.source.display()
        );
    }
    out
}

/// Render a descriptor for humans.
///
/// `api_key_present` only says whether the variable named by `api_key_env` is
/// set; its value is never read into this output.
#[must_use]
pub fn render_describe(descriptor: &ProviderDescriptor, api_key_present: bool) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let _ = writeln!(out, "provider: {}", descriptor.name);
    let _ = writeln!(out, "modality: {}", descriptor.modality.as_str());
    let _ = writeln!(out, "tier: descriptor");
    let _ = writeln!(out, "protocol_version: {}", descriptor.protocol_version);
    match &descriptor.adapter.endpoint {
        Some(endpoint) => {
            let _ = writeln!(
                out,
                "adapter: {} endpoint={endpoint}",
                adapter_kind(descriptor)
            );
        }
        None => {
            let _ = writeln!(out, "adapter: {}", adapter_kind(descriptor));
        }
    }
    match &descriptor.api_key_env {
        Some(variable) => {
            let state = if api_key_present {
                "<set>"
            } else {
                "<missing>"
            };
            let _ = writeln!(out, "api_key_env: {variable} {state}");
        }
        None => {
            let _ = writeln!(out, "api_key_env: <none>");
        }
    }
    let _ = writeln!(out, "capabilities: {}", joined(&descriptor.capabilities));
    let _ = writeln!(
        out,
        "provider_limits: max_concurrency={} max_inflight={}",
        optional(descriptor.provider_limits.max_concurrency),
        optional(descriptor.provider_limits.max_inflight)
    );
    let _ = writeln!(out, "models:");
    for model in &descriptor.models {
        let _ = writeln!(out, "  - id: {}", model.id);
        let _ = writeln!(out, "    display_name: {}", model.display_name);
        let _ = writeln!(out, "    capabilities: {}", joined(&model.capabilities));
        let _ = writeln!(
            out,
            "    limits: max_input_tokens={} sample_rates={:?} channels={:?}",
            optional(model.limits.max_input_tokens),
            model.limits.sample_rates,
            model.limits.channels
        );
    }
    let _ = writeln!(out, "config_schema:");
    out.push_str(&descriptor.config_schema_json());
    out.push('\n');
    out
}

fn adapter_kind(descriptor: &ProviderDescriptor) -> &str {
    let kind = descriptor.adapter.kind.trim();
    if kind.is_empty() {
        "<none>"
    } else {
        kind
    }
}

fn joined(items: &[String]) -> String {
    if items.is_empty() {
        "<none>".to_owned()
    } else {
        items.join(", ")
    }
}

fn optional(value: Option<u32>) -> String {
    value.map_or_else(|| "<unset>".to_owned(), |value| value.to_string())
}

/// Run one turn with the in-process fakes and render it.
///
/// # Errors
///
/// Returns a protocol-class error if a fake manifest or the session
/// configuration is rejected, and propagates any error from the turn.
pub async fn demo_fake() -> Result<(), ParlevoxError> {
    demo_fake_with(None).await
}

async fn demo_fake_with(record: Option<RecordOptions>) -> Result<(), ParlevoxError> {
    start_trace_export();
    let result = run_fake_demo(record).await;
    finish_trace_export();
    result
}

/// The one-turn fake body of [`demo_fake`], split out so the exporter is
/// flushed on both the success and the error path.
async fn run_fake_demo(record: Option<RecordOptions>) -> Result<(), ParlevoxError> {
    let registry = ProviderRegistry::fakes()?;
    let providers = registry.resolve(&ProviderNames::default())?;
    let session = build_session(SessionConfig::default(), providers)?;
    let session = attach_recorder(session, record.as_ref())?;
    println!("providers stt=fake-stt llm=fake-llm tts=fake-tts vad=fake-vad");
    run_and_render(session).await
}

/// Run the same turn with STT, LLM and TTS served by the Python sidecar.
///
/// # Errors
///
/// Returns a transport-class error when the bundled SDK cannot be located
/// (the message names `PARLEVOX_PYTHON_SDK` when that override is honoured but
/// invalid, and otherwise lists every path tried) or when the plugin cannot be
/// spawned or reaped, a protocol-class error for an invalid manifest or session
/// configuration, and any error raised by the turn.
pub async fn demo_sidecar() -> Result<(), ParlevoxError> {
    demo_sidecar_with(None).await
}

async fn demo_sidecar_with(record: Option<RecordOptions>) -> Result<(), ParlevoxError> {
    start_trace_export();
    let result = run_sidecar_demo(record).await;
    finish_trace_export();
    result
}

/// The one-turn sidecar body of [`demo_sidecar`], split out so the exporter is
/// flushed after the turn has rendered and after the child is reaped.
async fn run_sidecar_demo(record: Option<RecordOptions>) -> Result<(), ParlevoxError> {
    let sdk_path = resolve_sdk_path(&SdkSearch::from_process())
        .map_err(|err| ParlevoxError::Transport(err.to_string()))?;
    let client = Arc::new(SidecarClient::spawn(SidecarConfig::python_sdk_at(sdk_path)).await?);

    let (session, names) = build_sidecar_session(Arc::clone(&client))?;
    let session = attach_recorder(session, record.as_ref())?;
    println!(
        "providers stt={} llm={} tts={} vad={}",
        names.stt, names.llm, names.tts, names.vad
    );

    let result = run_and_render(session).await;
    // Always reap the child, even when the turn failed.
    let shutdown = client.shutdown().await;
    result?;
    shutdown
}

/// Run two deterministic turns through one [`SidecarPool`].
///
/// The first turn is rendered exactly as `demo --sidecar` renders it, so stdout
/// stays byte-identical to the single-shot demo. The second turn runs through
/// [`run_turn_events`] and prints nothing, which is what makes the reuse
/// observable only in the pool statistics. Those statistics go to stderr after
/// [`SidecarPool::shutdown`], so `shutdown` counts the real teardown rather
/// than a prediction of it.
///
/// The first lease is checked for the `reset` capability before anything is
/// written to stdout: without it a second lease would spawn a second process
/// while still looking like a pool. That case returns a protocol error naming
/// the missing capability instead of claiming reuse that did not happen.
///
/// # Errors
///
/// Returns a transport-class error when the bundled SDK cannot be located or
/// the plugin cannot be spawned or reaped, a protocol-class error when the
/// plugin does not advertise `reset` or a manifest or session configuration is
/// rejected, and any error raised by either turn. The pool is shut down before
/// returning in every case.
pub async fn demo_sidecar_pooled() -> Result<(), ParlevoxError> {
    demo_sidecar_pooled_with(None).await
}

async fn demo_sidecar_pooled_with(record: Option<RecordOptions>) -> Result<(), ParlevoxError> {
    let sdk_path = resolve_sdk_path(&SdkSearch::from_process())
        .map_err(|err| ParlevoxError::Transport(err.to_string()))?;
    let pool = Arc::new(SidecarPool::new(
        SidecarConfig::python_sdk_at(sdk_path),
        PoolConfig::default(),
    ));

    let turns = run_pooled_turns(Arc::clone(&pool), record.as_ref()).await;
    // Shut the pool down before reading stats so `shutdown` reflects the real
    // teardown, and do it on the failure path too.
    pool.shutdown().await;
    let stats = pool.stats();
    eprintln!(
        "pool spawned={} reused={} evicted={} shutdown={} reset_failures={} expired={} \
         release_refused={} retired={}",
        stats.spawned,
        stats.reused,
        stats.evicted,
        stats.shutdown,
        stats.reset_failures,
        stats.expired,
        stats.release_refused,
        stats.retired
    );
    turns
}

/// The two-leases-over-one-pool body of [`demo_sidecar_pooled`].
///
/// Split out so the caller owns shutdown and statistics for both outcomes.
async fn run_pooled_turns(
    pool: Arc<SidecarPool>,
    record: Option<&RecordOptions>,
) -> Result<(), ParlevoxError> {
    // Turn 1: the only turn that writes to stdout.
    let lease = pool.acquire().await?;
    if !lease.client().supports_reset() {
        return Err(ParlevoxError::Protocol(
            "the plugin does not advertise the reset capability, so it cannot be \
             reused by --pool"
                .into(),
        ));
    }
    let (session, names) = build_sidecar_session(Arc::clone(lease.client()))?;
    // A pooled run records only the first, rendered turn; the second is the
    // deliberately silent reuse probe and must not double-write the trace.
    let session = attach_recorder(session, record)?;
    println!(
        "providers stt={} llm={} tts={} vad={}",
        names.stt, names.llm, names.tts, names.vad
    );
    let first = run_and_render(session).await;
    let released = lease.release().await;
    first?;
    released?;

    // Turn 2: the same one-turn demo, deliberately silent.
    let lease = pool.acquire().await?;
    let (session, _names) = build_sidecar_session(Arc::clone(lease.client()))?;
    let second = run_turn_events(session).await;
    let released = lease.release().await;
    second?;
    released?;
    Ok(())
}

/// Drive one deterministic demo turn and return the report plus every event.
///
/// This is the event-producing entry point the parity test compares across the
/// fake and sidecar-adapted provider sets.
///
/// # Errors
///
/// Propagates the classified provider, protocol, transport, or closed-session
/// error from the turn.
pub async fn run_turn_events(
    mut session: Session,
) -> Result<(TurnReport, Vec<SessionEvent>), ParlevoxError> {
    let utterance = demo_utterance(16_000, 1, 40);
    let result = session.run_turn(utterance).await;
    let events = session.take_events();
    Ok((result?, events))
}

async fn run_and_render(mut session: Session) -> Result<(), ParlevoxError> {
    let utterance = demo_utterance(16_000, 1, 40);
    let result = session.run_turn(utterance).await;
    for event in session.take_events() {
        println!("{}", render(&event));
    }
    // Recording is opt-in, and the counters go to stderr so stdout stays
    // byte-identical to the unrecorded demo.
    print_recorder_stats(session.recorder_stats());
    match result {
        Ok(report) => {
            println!(
                "summary turn={} transcript={:?} reply={:?} audio_frames={} \
                 dropped_frames={} dropped_ms={} dropped_events={} \
                 sidecar_dropped_frames={} interrupted={} \
                 context_messages={} context_evicted_messages={}",
                report.turn,
                report.transcript,
                report.reply,
                report.audio_frames,
                report.dropped_frames,
                report.dropped_ms,
                report.dropped_events,
                report.sidecar_dropped_frames,
                report.interrupted,
                report.context_messages,
                report.context_evicted_messages
            );
            Ok(())
        }
        Err(err) => Err(err),
    }
}

pub fn render(event: &SessionEvent) -> String {
    match event {
        SessionEvent::TurnStarted { turn } => format!("event TurnStarted turn={turn}"),
        SessionEvent::TranscriptFinal { turn, text } => {
            format!("event TranscriptFinal turn={turn} text={text:?}")
        }
        SessionEvent::ReplyDelta { turn, text } => {
            format!("event ReplyDelta turn={turn} text={text:?}")
        }
        SessionEvent::AudioOut { turn, frame } => format!(
            "event AudioOut turn={turn} samples={} rate={} channels={}",
            frame.pcm.len(),
            frame.sample_rate,
            frame.channels
        ),
        SessionEvent::TurnDone {
            turn,
            audio_frames,
            dropped_frames,
            dropped_ms,
            dropped_events,
            sidecar_dropped_frames,
            interrupted,
        } => format!(
            "event TurnDone turn={turn} audio_frames={audio_frames} dropped_frames={dropped_frames} \
             dropped_ms={dropped_ms} dropped_events={dropped_events} \
             sidecar_dropped_frames={sidecar_dropped_frames} interrupted={interrupted}"
        ),
        SessionEvent::Failed { class, message } => {
            format!("event Failed class={class:?} message={message:?}")
        }
    }
}
