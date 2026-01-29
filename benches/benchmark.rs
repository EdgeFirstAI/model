use tracing::level_filters::LevelFilter;
use tracing_subscriber::{Layer, Registry, layer::SubscriberExt};

fn main() {
    let env_filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    let stdout_log = tracing_subscriber::fmt::layer()
        .pretty()
        .with_filter(env_filter);

    let env_filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    let journald = match tracing_journald::layer() {
        Ok(journald) => Some(journald.with_filter(env_filter)),
        Err(_) => None,
    };

    let subscriber = Registry::default().with(stdout_log).with(journald);
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
    tracing_log::LogTracer::init().unwrap();

    divan::main();
}
