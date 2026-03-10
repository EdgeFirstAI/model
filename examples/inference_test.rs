// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Standalone inference test tool.
//!
//! Loads a model (TFLite or DVM), runs inference on a known image, decodes
//! outputs through the same pipeline as the production service, and writes
//! an annotated output image with bounding boxes and instance segmentation
//! masks. Useful for validating model + decoder correctness without camera,
//! Zenoh, or any other system dependency.
//!
//! ```sh
//! inference_test --model yolov8n-seg.dvm --image zidane.jpg
//! inference_test --model yolov8n-seg.tflite --image zidane.jpg --output result.png
//! ```

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use edgefirst_hal::decoder::DecoderBuilder;
use edgefirst_hal::image::{PLANAR_RGB, PLANAR_RGB_INT8};
use edgefirst_hal::tensor::{TensorMapTrait, TensorTrait};
use edgefirst_model::{
    model::{ModelContext, decode_outputs, guess_model_config},
    runtime,
};
use image::{GenericImageView, RgbImage};

// ── Colour palette for drawing masks / boxes ────────────────────────────────

const PALETTE: &[(u8, u8, u8)] = &[
    (76, 175, 80),  // green
    (33, 150, 243), // blue
    (255, 152, 0),  // orange
    (156, 39, 176), // purple
    (0, 188, 212),  // cyan
    (244, 67, 54),  // red
    (255, 235, 59), // yellow
    (121, 85, 72),  // brown
    (233, 30, 99),  // pink
    (63, 81, 181),  // indigo
];

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "inference_test",
    about = "Run model inference on a static image"
)]
struct Args {
    /// Path to model file (.tflite or .dvm)
    #[arg(long)]
    model: PathBuf,

    /// Path to input image (JPEG/PNG)
    #[arg(long)]
    image: PathBuf,

    /// Path for annotated output image
    #[arg(long, default_value = "output.png")]
    output: PathBuf,

    /// Delegate / socket path (empty = default)
    #[arg(long, default_value = "")]
    delegate: String,

    /// Score threshold for detections
    #[arg(long, default_value_t = 0.25)]
    threshold: f32,

    /// IoU threshold for NMS
    #[arg(long, default_value_t = 0.45)]
    iou: f32,
}

// ── Drawing helpers ─────────────────────────────────────────────────────────

fn draw_rect(img: &mut RgbImage, x0: u32, y0: u32, x1: u32, y1: u32, color: [u8; 3]) {
    let (w, h) = img.dimensions();
    let x0 = x0.min(w.saturating_sub(1));
    let x1 = x1.min(w.saturating_sub(1));
    let y0 = y0.min(h.saturating_sub(1));
    let y1 = y1.min(h.saturating_sub(1));
    for x in x0..=x1 {
        img.put_pixel(x, y0, image::Rgb(color));
        img.put_pixel(x, y1, image::Rgb(color));
    }
    for y in y0..=y1 {
        img.put_pixel(x0, y, image::Rgb(color));
        img.put_pixel(x1, y, image::Rgb(color));
    }
}

fn blend_pixel(img: &mut RgbImage, x: u32, y: u32, color: (u8, u8, u8), alpha: f32) {
    let (w, h) = img.dimensions();
    if x >= w || y >= h {
        return;
    }
    let bg = img.get_pixel(x, y);
    let r = (bg[0] as f32 * (1.0 - alpha) + color.0 as f32 * alpha) as u8;
    let g = (bg[1] as f32 * (1.0 - alpha) + color.1 as f32 * alpha) as u8;
    let b = (bg[2] as f32 * (1.0 - alpha) + color.2 as f32 * alpha) as u8;
    img.put_pixel(x, y, image::Rgb([r, g, b]));
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() -> ExitCode {
    let args = Args::parse();

    // ── 1. Load model (same as main.rs) ─────────────────────────────────
    let mut runtime = match runtime::create_runtime(&args.model, &args.delegate) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Could not create runtime: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    let info = runtime.metadata().clone();

    let in_shape = runtime.input_shape(0);
    let in_h = in_shape.get(1).copied().unwrap_or(0);
    let in_w = in_shape.get(2).copied().unwrap_or(0);
    println!("Model input: {in_w}x{in_h}");
    println!("Labels: {:?}", info.labels);

    // ── 2. Build ModelContext (same as main.rs) ─────────────────────────
    let model_ctx = ModelContext {
        input_shapes: (0..runtime.input_count())
            .map(|i| runtime.input_shape(i).to_vec())
            .collect(),
        input_types: (0..runtime.input_count())
            .map(|i| runtime.input_dtype(i))
            .collect(),
        output_shapes: (0..runtime.output_count())
            .map(|i| runtime.output_shape(i).to_vec())
            .collect(),
        output_types: (0..runtime.output_count())
            .map(|i| runtime.output_dtype(i))
            .collect(),
        labels: info.labels.clone(),
        name: info.name.clone().unwrap_or_default(),
    };

    // ── 3. Build decoder (same as main.rs) ──────────────────────────────
    let mut decoder_builder = DecoderBuilder::new()
        .with_score_threshold(args.threshold)
        .with_iou_threshold(args.iou);

    if let Some(yaml) = &info.config_yaml {
        decoder_builder = decoder_builder.with_config_yaml_str(yaml.clone());
    } else {
        let input_dim = in_w.max(in_h) as f32;
        let output_quants: Vec<Option<(f32, i32)>> = (0..runtime.output_count())
            .map(|i| {
                runtime.output_quantization(i).map(|q| {
                    let is_box_output = model_ctx.output_shapes[i].contains(&4);
                    let scale = if is_box_output && input_dim > 1.0 {
                        q.scale / input_dim
                    } else {
                        q.scale
                    };
                    (scale, q.zero_point)
                })
            })
            .collect();

        let config = guess_model_config(&model_ctx.output_shapes, &output_quants);
        println!("Output shapes: {:?}", model_ctx.output_shapes);
        if let Some(cfg) = config {
            println!("Guessed config: {:?}", cfg);
            decoder_builder = decoder_builder.with_config(cfg);
        } else {
            eprintln!(
                "Could not guess model config from output shapes: {:?}",
                model_ctx.output_shapes
            );
            return ExitCode::FAILURE;
        }
    }

    let decoder = match decoder_builder.build() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Could not build decoder: {e:?}");
            return ExitCode::FAILURE;
        }
    };

    let model_type = decoder.model_type();
    let has_instance_seg = {
        use edgefirst_hal::decoder::configs::ModelType::*;
        matches!(
            model_type,
            YoloSegDet { .. }
                | YoloSplitSegDet { .. }
                | YoloEndToEndSegDet { .. }
                | YoloSplitEndToEndSegDet { .. }
        )
    };
    println!("Model type: {model_type:?}, instance_seg={has_instance_seg}");

    // ── 4. Load input image ─────────────────────────────────────────────
    let src_img = match image::open(&args.image) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Could not open image {}: {e}", args.image.display());
            return ExitCode::FAILURE;
        }
    };
    let (src_w, src_h) = src_img.dimensions();
    println!("Source image: {src_w}x{src_h}");

    // Resize to model input dimensions
    let resized = src_img.resize_exact(
        in_w as u32,
        in_h as u32,
        image::imageops::FilterType::Triangle,
    );
    let rgb = resized.to_rgb8();

    // ── 5. Copy to input tensor (same logic as main.rs) ─────────────────
    let input_fourcc = runtime.input_fourcc(0);
    let is_planar = matches!(input_fourcc, f if f == PLANAR_RGB || f == PLANAR_RGB_INT8);

    {
        let pixels = rgb.as_raw();
        let input = runtime.input_tensor(0);
        let mut dst_map = match input.map() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Could not map input tensor: {e:?}");
                return ExitCode::FAILURE;
            }
        };
        let dst = dst_map.as_mut_slice();
        if is_planar {
            let plane_size = in_w * in_h;
            let xor_mask = if input_fourcc == PLANAR_RGB_INT8 {
                0x80u8
            } else {
                0x00u8
            };
            for i in 0..plane_size {
                dst[i] = pixels[i * 3] ^ xor_mask;
                dst[plane_size + i] = pixels[i * 3 + 1] ^ xor_mask;
                dst[2 * plane_size + i] = pixels[i * 3 + 2] ^ xor_mask;
            }
        } else {
            dst[..pixels.len()].copy_from_slice(pixels);
        }
    }

    // ── 6. Run inference (same as main.rs) ──────────────────────────────
    let timing = match runtime.invoke() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Inference failed: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "Inference: input={:?} model={:?} output={:?}",
        timing.input_time, timing.model_time, timing.output_time,
    );

    // ── 7. Decode outputs (same as main.rs) ─────────────────────────────
    let mut output_boxes = Vec::with_capacity(50);
    let mut output_masks = Vec::with_capacity(50);

    if let Err(e) = decode_outputs(
        runtime.as_ref(),
        &decoder,
        &mut output_boxes,
        &mut output_masks,
    ) {
        eprintln!("Decode failed: {e:?}");
        return ExitCode::FAILURE;
    }

    println!("Detections: {}", output_boxes.len());
    println!("Masks: {}", output_masks.len());

    // ── 8. Print mask diagnostics ───────────────────────────────────────
    for (i, seg) in output_masks.iter().enumerate() {
        let shape = seg.segmentation.shape();
        let data = seg.segmentation.as_slice().unwrap_or(&[]);
        let min = data.iter().copied().min().unwrap_or(0);
        let max = data.iter().copied().max().unwrap_or(0);
        let sum: u64 = data.iter().map(|&v| v as u64).sum();
        let mean = if data.is_empty() {
            0.0
        } else {
            sum as f64 / data.len() as f64
        };
        let above_128 = data.iter().filter(|&&v| v >= 128).count();
        let total = data.len();
        let pct = if total > 0 {
            above_128 as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "  Mask {i}: shape={shape:?} min={min} max={max} mean={mean:.1} \
             above_128={above_128}/{total} ({pct:.1}%)",
        );
    }

    // ── 9. Draw results on source image ─────────────────────────────────
    let mut canvas = src_img.to_rgb8();
    let (cw, ch) = canvas.dimensions();

    // Draw instance masks first (underneath boxes)
    if has_instance_seg {
        for (i, (b, seg)) in output_boxes.iter().zip(output_masks.iter()).enumerate() {
            let color = PALETTE[i % PALETTE.len()];
            let mask_shape = seg.segmentation.shape();
            let mask_h = mask_shape[0];
            let mask_w = mask_shape[1];

            // Bounding box in pixel coords (boxes are normalised [0,1])
            let bx0 = (b.bbox.xmin * cw as f32) as u32;
            let by0 = (b.bbox.ymin * ch as f32) as u32;
            let bx1 = (b.bbox.xmax * cw as f32) as u32;
            let by1 = (b.bbox.ymax * ch as f32) as u32;
            let box_w = bx1.saturating_sub(bx0).max(1);
            let box_h = by1.saturating_sub(by0).max(1);

            for my in 0..mask_h {
                for mx in 0..mask_w {
                    let val = seg.segmentation[[my, mx, 0]];
                    if val >= 128 {
                        // Map mask pixel to canvas pixel
                        let px = bx0 + (mx as u32 * box_w) / mask_w as u32;
                        let py = by0 + (my as u32 * box_h) / mask_h as u32;
                        // Use sigmoid value for alpha: scale 128-255 → 0.2-0.5
                        let alpha = 0.2 + (val as f32 - 128.0) / 127.0 * 0.3;
                        blend_pixel(&mut canvas, px, py, color, alpha);
                    }
                }
            }
        }
    }

    // Draw bounding boxes and labels
    for (i, b) in output_boxes.iter().enumerate() {
        let color = PALETTE[i % PALETTE.len()];
        let rgb_color = [color.0, color.1, color.2];

        let x0 = (b.bbox.xmin * cw as f32) as u32;
        let y0 = (b.bbox.ymin * ch as f32) as u32;
        let x1 = (b.bbox.xmax * cw as f32) as u32;
        let y1 = (b.bbox.ymax * ch as f32) as u32;

        // Draw box with 2px thickness
        draw_rect(&mut canvas, x0, y0, x1, y1, rgb_color);
        if x0 > 0 && y0 > 0 {
            draw_rect(
                &mut canvas,
                x0.saturating_sub(1),
                y0.saturating_sub(1),
                (x1 + 1).min(cw - 1),
                (y1 + 1).min(ch - 1),
                rgb_color,
            );
        }

        let label = info
            .labels
            .get(b.label)
            .cloned()
            .unwrap_or_else(|| b.label.to_string());
        println!(
            "  [{i}] {label} score={:.3} box=({:.3},{:.3})-({:.3},{:.3})",
            b.score, b.bbox.xmin, b.bbox.ymin, b.bbox.xmax, b.bbox.ymax,
        );
    }

    // ── 10. Save output ─────────────────────────────────────────────────
    match canvas.save(&args.output) {
        Ok(()) => println!("Saved: {}", args.output.display()),
        Err(e) => {
            eprintln!("Could not save output: {e}");
            return ExitCode::FAILURE;
        }
    }

    ExitCode::SUCCESS
}
