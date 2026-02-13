#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

bindgen --dynamic-loading g2d --allowlist-function 'g2d_.*' g2d.h > src/ffi.rs