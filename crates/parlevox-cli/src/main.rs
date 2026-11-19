//! Thin binary wrapper over the testable `parlevox-cli` library.

#[tokio::main]
async fn main() {
    if let Err(err) = parlevox_cli::run_args().await {
        eprintln!("parlevox: {err}");
        std::process::exit(1);
    }
}
