//! The descriptors shipped under `config/providers` must parse and validate.

use std::path::PathBuf;

use parlevox_providers::{load_dir, ProviderDescriptor};

fn descriptors_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("config")
        .join("providers")
}

#[test]
fn shipped_descriptors_parse_and_validate() {
    let dir = descriptors_dir();
    let paths = load_dir(&dir).expect("config/providers is readable");
    assert_eq!(paths.len(), 3, "three shipped descriptors");

    let mut names = Vec::with_capacity(paths.len());
    for path in &paths {
        let descriptor = ProviderDescriptor::from_path(path)
            .unwrap_or_else(|err| panic!("{} should parse: {err}", path.display()));
        descriptor
            .validate()
            .unwrap_or_else(|err| panic!("{} should validate: {err}", path.display()));
        assert_eq!(descriptor.api_key_env.as_deref(), Some("OPENAI_API_KEY"));
        assert!(
            !descriptor.models.is_empty(),
            "{} declares at least one model",
            descriptor.name
        );
        names.push(descriptor.name);
    }
    names.sort();
    assert_eq!(names, vec!["openai-llm", "openai-stt", "openai-tts"]);
}
