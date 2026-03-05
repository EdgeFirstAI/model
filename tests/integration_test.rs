// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Integration tests for EdgeFirst Model service.
//!
//! These tests require real hardware (NXP i.MX8M Plus with NPU and camera)
//! and are marked with `#[ignore]`. They are run on the `imx8mpevk` hardware
//! runner in CI.
//!
//! The test launches the edgefirst-model service, subscribes to model output
//! topics, verifies messages are received and decodable, then sends SIGTERM
//! for graceful shutdown.

use edgefirst_schemas::{edgefirst_msgs::Model, serde_cdr};
use std::{
    env,
    process::{Child, Command},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
use zenoh::Wait;

/// Expected minimum message rate (Hz) from the model service.
/// Model typically runs at 15-30 fps; 10 Hz is very conservative.
const MIN_EXPECTED_RATE_HZ: f64 = 10.0;

/// Duration to collect model messages before analyzing.
const COLLECTION_DURATION: Duration = Duration::from_secs(10);

/// Time to wait for the model service to start up.
/// Model loading (TFLite + delegate + first inference warmup) is much
/// slower than simpler services like the IMU.
const STARTUP_DELAY: Duration = Duration::from_secs(10);

/// Topic the model service publishes unified output to.
const MODEL_OUTPUT_TOPIC: &str = "rt/model/output";

/// Find the edgefirst-model binary.
/// In CI, it's passed via environment variable. Locally, look in target directory.
fn find_model_binary() -> String {
    if let Ok(path) = env::var("MODEL_BINARY") {
        return path;
    }

    // Try common locations
    let candidates = [
        "target/llvm-cov-target/profiling/edgefirst-model",
        "target/profiling/edgefirst-model",
        "target/release/edgefirst-model",
        "target/debug/edgefirst-model",
    ];

    for candidate in candidates {
        if std::path::Path::new(candidate).exists() {
            return candidate.to_string();
        }
    }

    panic!("Could not find edgefirst-model binary. Set MODEL_BINARY environment variable.");
}

/// Start the model service as a child process.
///
/// The model path must be provided via the `MODEL` environment variable.
/// Tracy is explicitly disabled to avoid interference with coverage tests.
fn start_model_service() -> Child {
    let binary = find_model_binary();
    println!("Starting model service: {binary}");

    let model_path = env::var("MODEL")
        .expect("MODEL environment variable must be set to the path of a TFLite model file");

    let mut cmd = Command::new(&binary);
    cmd.arg("--model").arg(&model_path);

    // Disable Tracy to avoid interference with coverage
    cmd.env("TRACY", "false");

    // Pass through delegate path if set
    if let Ok(delegate) = env::var("DELEGATE") {
        cmd.arg("--delegate").arg(&delegate);
    }

    cmd.spawn().expect("Failed to start model service")
}

/// Send SIGTERM to gracefully stop the model service.
fn stop_model_service(mut child: Child) {
    println!("Sending SIGTERM to model service (pid: {})", child.id());

    unsafe {
        libc::kill(child.id() as i32, libc::SIGTERM);
    }

    // Wait for graceful shutdown (up to 5 seconds)
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                println!("Model service exited with status: {status:?}");
                return;
            }
            Ok(None) => {
                if start.elapsed() > Duration::from_secs(5) {
                    println!("Model service did not exit gracefully, killing...");
                    let _ = child.kill();
                    return;
                }
                thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                println!("Error waiting for model service: {e}");
                return;
            }
        }
    }
}

/// Integration test for model inference output.
///
/// This test:
/// 1. Starts the edgefirst-model service
/// 2. Waits for model loading and first inference warmup
/// 3. Subscribes to the unified model output topic
/// 4. Collects messages for COLLECTION_DURATION
/// 5. Verifies messages are valid Model messages with non-zero timing
/// 6. Checks the publishing rate meets minimum threshold
/// 7. Gracefully stops the model service
#[test]
#[ignore] // Requires hardware - run on imx8mpevk runner
fn test_model_inference() {
    // Start the model service
    let model_process = start_model_service();

    // Give the service time to load model and start inference
    println!("Waiting {STARTUP_DELAY:?} for model startup...");
    thread::sleep(STARTUP_DELAY);

    // Open Zenoh session
    let session = zenoh::open(zenoh::Config::default())
        .wait()
        .expect("Failed to open Zenoh session");

    // Subscribe to model output topic
    let message_count = Arc::new(AtomicU64::new(0));
    let message_count_clone = message_count.clone();

    let subscriber = session
        .declare_subscriber(MODEL_OUTPUT_TOPIC)
        .callback(move |sample| {
            // Try to decode the message
            match serde_cdr::deserialize::<Model>(&sample.payload().to_bytes()) {
                Ok(model) => {
                    // Verify timing fields are non-zero
                    let input_ns = model.input_time.sec as u64 * 1_000_000_000
                        + model.input_time.nanosec as u64;
                    let model_ns = model.model_time.sec as u64 * 1_000_000_000
                        + model.model_time.nanosec as u64;
                    let decode_ns = model.decode_time.sec as u64 * 1_000_000_000
                        + model.decode_time.nanosec as u64;

                    if input_ns > 0 && model_ns > 0 && decode_ns > 0 {
                        message_count_clone.fetch_add(1, Ordering::SeqCst);
                    } else {
                        eprintln!(
                            "Model message has zero timing: input={input_ns}ns model={model_ns}ns decode={decode_ns}ns"
                        );
                    }
                }
                Err(e) => {
                    eprintln!("Failed to decode Model message: {e}");
                }
            }
        })
        .wait()
        .expect("Failed to create subscriber");

    println!("Collecting model messages for {COLLECTION_DURATION:?}...");
    thread::sleep(COLLECTION_DURATION);

    // Get final count
    let count = message_count.load(Ordering::SeqCst);
    let rate = count as f64 / COLLECTION_DURATION.as_secs_f64();

    println!("Received {count} messages in {COLLECTION_DURATION:?}");
    println!("Message rate: {rate:.1} Hz");

    // Clean up
    drop(subscriber);
    drop(session);
    stop_model_service(model_process);

    // Assertions
    assert!(count > 0, "No model output messages received!");
    assert!(
        rate >= MIN_EXPECTED_RATE_HZ,
        "Model rate {rate:.1} Hz is below minimum {MIN_EXPECTED_RATE_HZ:.1} Hz",
    );

    println!("Integration test passed!");
}

/// Test that the model service handles SIGTERM gracefully.
#[test]
#[ignore] // Requires hardware - run on imx8mpevk runner
fn test_graceful_shutdown() {
    // Start the model service
    let model_process = start_model_service();

    // Give the service time to initialize and start publishing
    println!("Waiting {STARTUP_DELAY:?} for model startup...");
    thread::sleep(STARTUP_DELAY);

    // Send SIGTERM
    let pid = model_process.id();
    println!("Sending SIGTERM to model service (pid: {pid})");
    unsafe {
        libc::kill(pid as i32, libc::SIGTERM);
    }

    // Wait for exit with timeout
    let mut child = model_process;
    let start = Instant::now();
    let exit_status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if start.elapsed() > Duration::from_secs(5) {
                    println!("Timeout waiting for graceful shutdown");
                    let _ = child.kill();
                    break None;
                }
                thread::sleep(Duration::from_millis(100));
            }
            Err(_) => break None,
        }
    };

    // Verify it exited cleanly
    assert!(
        exit_status.is_some(),
        "Model service did not exit within timeout"
    );

    let status = exit_status.unwrap();
    println!("Model service exited with status: {status:?}");

    // On Unix, SIGTERM results in exit code 0 if handled properly
    // or signal termination if not
    assert!(
        status.success() || status.code().is_none(),
        "Model service did not exit cleanly: {status:?}",
    );

    println!("Graceful shutdown test passed!");
}
