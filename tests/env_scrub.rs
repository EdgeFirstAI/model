// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! End-to-end check that `KEY=""` in the environment behaves as unset.
//!
//! Runs with `harness = false` so this `main` is the only thread in the
//! process when the environment is mutated, which `scrub_empty_env` requires.
//!
//! `main` speaks the small subset of the libtest CLI that `cargo test` and
//! `cargo nextest` use to enumerate (`--list --format terse`) and select
//! (`--exact <name>`, `--ignored`, `--skip <pattern>`, positional filters)
//! tests, so the target is discovered and reported like any other test.

use clap::Parser;
use edgefirst_model::args::{Args, KEEP, scrub_empty_env};

/// The single test this binary provides, as reported to the harness.
const TEST_NAME: &str = "empty_env_is_treated_as_unset";

/// Float, boolean, integer and path arguments, all written as `KEY=""` in
/// /etc/default/model.
const VARS: [&str; 4] = ["THRESHOLD", "TRACK", "MAX_BOXES", "EDGEFIRST_CONFIG"];
const ARGV: [&str; 3] = ["edgefirst-model", "--model", "x"];

/// libtest flags that consume the following argument, so it is not a filter.
const VALUE_FLAGS: [&str; 5] = [
    "--test-threads",
    "--format",
    "--logfile",
    "--color",
    "--shuffle-seed",
];

/// What the harness asked this binary to do.
struct Request {
    list: bool,
    ignored: bool,
    exact: bool,
    filters: Vec<String>,
    skips: Vec<String>,
}

fn parse_request(argv: impl IntoIterator<Item = String>) -> Request {
    let mut req = Request {
        list: false,
        ignored: false,
        exact: false,
        filters: Vec::new(),
        skips: Vec::new(),
    };
    let mut argv = argv.into_iter();
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--list" => req.list = true,
            "--ignored" => req.ignored = true,
            "--exact" => req.exact = true,
            "--skip" => req.skips.extend(argv.next()),
            flag if VALUE_FLAGS.contains(&flag) => {
                argv.next();
            }
            flag if flag.starts_with('-') => {
                if let Some(pattern) = flag.strip_prefix("--skip=") {
                    req.skips.push(pattern.to_owned());
                }
            }
            filter => req.filters.push(filter.to_owned()),
        }
    }
    req
}

/// libtest matching: substring by default, equality under `--exact`.
fn matches(req: &Request, pattern: &str) -> bool {
    if req.exact {
        pattern == TEST_NAME
    } else {
        TEST_NAME.contains(pattern)
    }
}

fn selected(req: &Request) -> bool {
    let filtered_in = req.filters.is_empty() || req.filters.iter().any(|f| matches(req, f));
    filtered_in && !req.skips.iter().any(|s| matches(req, s))
}

fn main() {
    let req = parse_request(std::env::args().skip(1));
    // This binary has no #[ignore]d tests, so `--ignored` selects nothing.
    if req.list {
        if !req.ignored && selected(&req) {
            println!("{TEST_NAME}: test");
        }
        return;
    }
    if req.ignored || !selected(&req) {
        return;
    }

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
