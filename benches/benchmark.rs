use edgefirst_model::model::{fast_sigmoid, sigmoid};
use rand::{rng, seq::SliceRandom};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::{Layer, Registry, layer::SubscriberExt};

#[divan::bench]
fn sigmoid_bench(bencher: divan::Bencher) {
    let cap = 50 * 50 * 54;
    let mut rng = rng();
    bencher
        .with_inputs(|| {
            let mut nums: Vec<f32> = Vec::with_capacity(cap);
            for i in 0..cap {
                nums.push((i as f32 / cap as f32 - 0.5) * 6.0);
            }
            nums.shuffle(&mut rng);
            nums
        })
        .bench_local_refs(|nums| {
            nums.iter_mut().for_each(|x| *x = sigmoid(*x));
        });
}

#[divan::bench]
fn sigmoid_fast_bench(bencher: divan::Bencher) {
    let cap = 50 * 50 * 54;
    let mut rng = rng();
    bencher
        .with_inputs(|| {
            let mut nums: Vec<f32> = Vec::with_capacity(cap);
            for i in 0..cap {
                nums.push((i as f32 / cap as f32 - 0.5) * 6.0);
            }
            nums.shuffle(&mut rng);
            nums
        })
        .bench_local_refs(|nums| {
            nums.iter_mut().for_each(|x| *x = fast_sigmoid(*x));
        });
}

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
