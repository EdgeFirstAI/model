// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! End-to-end check that `KEY=""` in the environment behaves as unset.
//!
//! Runs with `harness = false` so this `main` is the only thread in the
//! process when the environment is mutated, which `scrub_empty_env` requires.

use clap::Parser;
use edgefirst_model::args::{Args, KEEP, scrub_empty_env};

const ARGV: [&str; 3] = ["edgefirst-model", "--model", "x"];
const VARS: [&str; 4] = ["THRESHOLD", "TRACK", "MAX_BOXES", "EDGEFIRST_CONFIG"];

fn main() {
    for name in VARS {
        // SAFETY: single-threaded — this is `main` before any thread is spawned.
        unsafe { std::env::set_var(name, "") };
    }
    let before = Args::try_parse_from(ARGV);
    assert!(
        before.is_err(),
        "empty vars must fail to parse before scrubbing: {before:?}"
    );

    // SAFETY: still single-threaded.
    unsafe { scrub_empty_env::<Args>(KEEP) };
    for name in VARS {
        assert!(
            std::env::var_os(name).is_none(),
            "{name} should have been removed"
        );
    }
    let args = Args::try_parse_from(ARGV).expect("defaults must apply after scrubbing");
    assert_eq!(args.threshold, 0.45);
    assert!(!args.track);
    assert_eq!(args.max_boxes, 100);
    assert_eq!(args.edgefirst_config(), None);

    // A model is required by design: MODEL="" is scrubbed to unset and clap
    // must then report the missing argument rather than accept "".
    // SAFETY: still single-threaded.
    unsafe { std::env::set_var("MODEL", "") };
    // SAFETY: as above.
    unsafe { scrub_empty_env::<Args>(KEEP) };
    assert!(
        std::env::var_os("MODEL").is_none(),
        "MODEL should have been removed"
    );
    let err = Args::try_parse_from(["edgefirst-model"]).expect_err("MODEL=\"\" must not parse");
    assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);

    println!("env_scrub: ok");
}
