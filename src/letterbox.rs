// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Aspect-preserving letterbox transform.
//!
//! The camera frame and the model input rarely share an aspect ratio. Instead
//! of stretching (which distorts objects), the preprocessing step scales the
//! frame to fit and pads the remainder with a neutral grey — a *letterbox*.
//!
//! The model then sees a padded canvas, so its decoded coordinates are
//! relative to that canvas, not the camera frame. [`LetterboxTransform`]
//! carries the geometry needed both to drive the HAL `ImageProcessor` (via
//! [`LetterboxTransform::crop`]) and to map decoded boxes/masks back to
//! camera-frame coordinates afterwards.

use edgefirst_hal::decoder::{BoundingBox, Segmentation};
use edgefirst_hal::image::Crop;
use ndarray::s;

/// Padding fill colour for the letterbox bars (RGBA). Matches the neutral grey
/// used by Ultralytics YOLO training preprocessing.
const PAD_COLOR: [u8; 4] = [114, 114, 114, 255];

/// Geometry of an aspect-preserving fit of a source image into a destination
/// canvas.
#[derive(Debug, Clone, Copy)]
pub struct LetterboxTransform {
    /// Destination canvas width (model input width).
    dst_w: usize,
    /// Destination canvas height (model input height).
    dst_h: usize,
    /// Width of the scaled source content inside the canvas.
    scaled_w: usize,
    /// Height of the scaled source content inside the canvas.
    scaled_h: usize,
    /// Left padding (grey bar width) in canvas pixels.
    pad_x: usize,
    /// Top padding (grey bar height) in canvas pixels.
    pad_y: usize,
}

impl LetterboxTransform {
    /// Compute the transform fitting a `src_w × src_h` image into a
    /// `dst_w × dst_h` canvas while preserving aspect ratio.
    pub fn compute(src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Self {
        if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
            return Self {
                dst_w: dst_w.max(1),
                dst_h: dst_h.max(1),
                scaled_w: dst_w.max(1),
                scaled_h: dst_h.max(1),
                pad_x: 0,
                pad_y: 0,
            };
        }

        // Match HAL `letterbox_rect` so Crop::letterbox and unletter share
        // the same placement (avoids 1px pad/scaled drift across formulas).
        let src_aspect = src_w as f64 / src_h as f64;
        let dst_aspect = dst_w as f64 / dst_h as f64;
        let (scaled_w, scaled_h) = if src_aspect > dst_aspect {
            (dst_w, ((dst_w as f64 / src_aspect).round() as usize).max(1))
        } else {
            (((dst_h as f64 * src_aspect).round() as usize).max(1), dst_h)
        };
        let pad_x = dst_w.saturating_sub(scaled_w) / 2;
        let pad_y = dst_h.saturating_sub(scaled_h) / 2;

        Self {
            dst_w,
            dst_h,
            scaled_w,
            scaled_h,
            pad_x,
            pad_y,
        }
    }

    /// Whether this transform pads at all (false when aspect ratios match).
    pub fn is_padded(&self) -> bool {
        self.pad_x != 0 || self.pad_y != 0
    }

    /// HAL [`Crop`] placing the scaled source content centered in the canvas,
    /// with the surrounding letterbox bars filled with [`PAD_COLOR`].
    ///
    /// Always uses [`Crop::letterbox`] so placement matches
    /// [`LetterboxTransform::compute`] (same HAL `letterbox_rect` math).
    pub fn crop(&self) -> Crop {
        Crop::letterbox(PAD_COLOR)
    }

    /// Map a canvas-normalized x coordinate back to source-normalized `[0,1]`.
    fn unletter_x(&self, x: f32) -> f32 {
        ((x * self.dst_w as f32 - self.pad_x as f32) / self.scaled_w as f32).clamp(0.0, 1.0)
    }

    /// Map a canvas-normalized y coordinate back to source-normalized `[0,1]`.
    fn unletter_y(&self, y: f32) -> f32 {
        ((y * self.dst_h as f32 - self.pad_y as f32) / self.scaled_h as f32).clamp(0.0, 1.0)
    }

    /// Rewrite a decoder bounding box from canvas-normalized to
    /// source-normalized coordinates.
    pub fn unletter_box(&self, bbox: &mut BoundingBox) {
        bbox.xmin = self.unletter_x(bbox.xmin);
        bbox.xmax = self.unletter_x(bbox.xmax);
        bbox.ymin = self.unletter_y(bbox.ymin);
        bbox.ymax = self.unletter_y(bbox.ymax);
    }

    /// Rewrite an instance-segmentation mask's region bounds from
    /// canvas-normalized to source-normalized coordinates. The mask pixel
    /// data is unchanged — it describes content inside the (now corrected)
    /// region.
    pub fn unletter_instance_mask(&self, seg: &mut Segmentation) {
        seg.xmin = self.unletter_x(seg.xmin);
        seg.xmax = self.unletter_x(seg.xmax);
        seg.ymin = self.unletter_y(seg.ymin);
        seg.ymax = self.unletter_y(seg.ymax);
    }

    /// Crop a full-canvas semantic-segmentation mask to the letterbox content
    /// region so it corresponds 1:1 with the camera frame.
    ///
    /// A semantic mask spans the whole padded canvas; the grey-bar rows and
    /// columns carry no real classification. This slices them off and resets
    /// the region bounds to the full frame.
    pub fn crop_semantic_mask(&self, seg: &mut Segmentation) {
        if !self.is_padded() {
            return;
        }
        let shape = seg.segmentation.shape();
        let (mask_h, mask_w) = (shape[0], shape[1]);
        if mask_h == 0 || mask_w == 0 {
            return;
        }

        // Map the canvas content rectangle into mask-array index space.
        let to_idx = |num: usize, den: usize, extent: usize| -> usize {
            (((num as f64 / den as f64) * extent as f64).round() as usize).min(extent)
        };
        let x0 = to_idx(self.pad_x, self.dst_w, mask_w);
        let x1 = to_idx(self.pad_x + self.scaled_w, self.dst_w, mask_w).max(x0 + 1);
        let y0 = to_idx(self.pad_y, self.dst_h, mask_h);
        let y1 = to_idx(self.pad_y + self.scaled_h, self.dst_h, mask_h).max(y0 + 1);
        let x1 = x1.min(mask_w);
        let y1 = y1.min(mask_h);

        seg.segmentation = seg.segmentation.slice(s![y0..y1, x0..x1, ..]).to_owned();
        seg.xmin = 0.0;
        seg.ymin = 0.0;
        seg.xmax = 1.0;
        seg.ymax = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_always_uses_letterbox_fit() {
        let padded = LetterboxTransform::compute(1280, 720, 640, 640);
        assert!(padded.is_padded());
        let crop = padded.crop();
        assert!(matches!(
            crop.fit,
            edgefirst_hal::image::Fit::Letterbox { pad: PAD_COLOR }
        ));

        let square = LetterboxTransform::compute(640, 640, 640, 640);
        assert!(!square.is_padded());
        let crop = square.crop();
        assert!(matches!(
            crop.fit,
            edgefirst_hal::image::Fit::Letterbox { .. }
        ));
    }

    #[test]
    fn matches_hal_letterbox_rect_on_odd_aspect() {
        // Aspect-driven HAL formula can differ from min-scale rounding.
        let lb = LetterboxTransform::compute(85, 320, 64, 224);
        assert_eq!(lb.scaled_h, 224);
        assert_eq!(
            lb.scaled_w,
            ((224.0_f64 * (85.0 / 320.0)).round() as usize).max(1)
        );
        assert_eq!(lb.pad_x, (64 - lb.scaled_w) / 2);
        assert_eq!(lb.pad_y, 0);
    }

    #[test]
    fn square_into_square_is_noop() {
        let lb = LetterboxTransform::compute(640, 640, 320, 320);
        assert!(!lb.is_padded());
        let mut b = BoundingBox::new(0.25, 0.25, 0.75, 0.75);
        lb.unletter_box(&mut b);
        assert!((b.xmin - 0.25).abs() < 1e-4);
        assert!((b.ymax - 0.75).abs() < 1e-4);
    }

    #[test]
    fn wide_source_pads_top_and_bottom() {
        // 1280x720 into 640x640 -> scale 0.5 -> 640x360, pad_y = 140.
        let lb = LetterboxTransform::compute(1280, 720, 640, 640);
        assert!(lb.is_padded());
        assert_eq!(lb.scaled_w, 640);
        assert_eq!(lb.scaled_h, 360);
        assert_eq!(lb.pad_y, 140);

        // A box spanning the content vertically (canvas y 140..500) maps to
        // the full source height.
        let mut b = BoundingBox::new(0.0, 140.0 / 640.0, 1.0, 500.0 / 640.0);
        lb.unletter_box(&mut b);
        assert!(b.ymin.abs() < 1e-3, "ymin={}", b.ymin);
        assert!((b.ymax - 1.0).abs() < 1e-3, "ymax={}", b.ymax);
    }

    #[test]
    fn tall_source_pads_left_and_right() {
        // 720x1280 into 640x640 -> scale 0.5 -> 360x640, pad_x = 140.
        let lb = LetterboxTransform::compute(720, 1280, 640, 640);
        assert!(lb.is_padded());
        assert_eq!(lb.scaled_w, 360);
        assert_eq!(lb.scaled_h, 640);
        assert_eq!(lb.pad_x, 140);

        // Content-x [140, 500] (canvas) maps to source [0, 1].
        let mut b = BoundingBox::new(140.0 / 640.0, 0.0, 500.0 / 640.0, 1.0);
        lb.unletter_box(&mut b);
        assert!(b.xmin.abs() < 1e-3, "xmin={}", b.xmin);
        assert!((b.xmax - 1.0).abs() < 1e-3, "xmax={}", b.xmax);

        // A box entirely in the left pad clamps to xmin=xmax=0.
        let mut pad = BoundingBox::new(0.0, 0.1, 0.05, 0.9);
        lb.unletter_box(&mut pad);
        assert_eq!(pad.xmin, 0.0);
        assert_eq!(pad.xmax, 0.0);
    }
}
