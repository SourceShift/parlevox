//! Thin binary wrapper over the testable `voicekit-cli` library.

#[tokio::main]
async fn main() {
    if let Err(err) = voicekit_cli::run_args().await {
        eprintln!("voicekit: {err}");
        std::process::exit(1);
    }
}
