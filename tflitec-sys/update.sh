#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

BINDGEN_EXTRA_CLANG_ARGS="-I./" bindgen --dynamic-loading tensorflowlite_c --wrap-unsafe-ops --allowlist-function 'TfLite.*' wrapper.h > src/ffi.rs
flatc --rust ./metadata_schema.fbs && mv metadata_schema_generated.rs ./src/metadata_schema_generated.rs
flatc --rust ./tensorflow/compiler/mlir/lite/schema/schema.fbs && mv schema_generated.rs ./src/schema_generated.rs
cargo clippy --fix --allow-dirty
