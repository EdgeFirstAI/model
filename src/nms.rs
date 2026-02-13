// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

use std::time::Instant;

use log::trace;
use ndarray::{
    ArrayView1, ArrayView2, Zip,
    parallel::prelude::{IntoParallelIterator, ParallelIterator},
};

use crate::model::DetectBox;

pub fn decode_boxes_and_nms(
    score_threshold: f32,
    iou_threshold: f32,
    scores_tensor: ArrayView2<f32>,
    boxes_tensor: ArrayView2<f32>,
    mask_coeff: Option<ArrayView2<f32>>,
    output_boxes: &mut Vec<DetectBox>,
    ignore_class: bool,
) {
    let start = Instant::now();
    let boxes = decode_boxes(score_threshold, scores_tensor, boxes_tensor, mask_coeff);
    let boxes = nms(iou_threshold, boxes, ignore_class);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.into_iter().take(len) {
        output_boxes.push(b);
    }
    trace!("Box decode and nms takes {:?}", start.elapsed());
}

pub fn decode_boxes(
    threshold: f32,
    scores: ArrayView2<f32>,
    boxes: ArrayView2<f32>,
    mask_coeff: Option<ArrayView2<f32>>,
) -> Vec<DetectBox> {
    assert_eq!(scores.dim().0, boxes.dim().0);
    assert_eq!(boxes.dim().1, 4);
    if let Some(mask_coeff) = mask_coeff {
        Zip::from(scores.rows())
            .and(boxes.rows())
            .and(mask_coeff.rows())
            .into_par_iter()
            .filter_map(|(score, bbox, mask)| {
                let (score_, label) = arg_max(score);
                if score_ < threshold {
                    return None;
                }

                Some(DetectBox {
                    label,
                    score: score_,
                    xmin: bbox[0],
                    ymin: bbox[1],
                    xmax: bbox[2],
                    ymax: bbox[3],
                    mask_coeff: Some(mask.to_owned()),
                })
            })
            .collect()
    } else {
        Zip::from(scores.rows())
            .and(boxes.rows())
            .into_par_iter()
            .filter_map(|(score, bbox)| {
                let (score_, label) = arg_max(score);
                if score_ < threshold {
                    return None;
                }

                Some(DetectBox {
                    label,
                    score: score_,
                    xmin: bbox[0],
                    ymin: bbox[1],
                    xmax: bbox[2],
                    ymax: bbox[3],
                    mask_coeff: None,
                })
            })
            .collect()
    }
}

fn arg_max<T: PartialOrd + Copy>(score: ArrayView1<T>) -> (T, usize) {
    score
        .iter()
        .enumerate()
        .fold((score[0], 0), |(max, arg_max), (ind, s)| {
            if max > *s { (max, arg_max) } else { (*s, ind) }
        })
}

pub fn nms(iou: f32, boxes: Vec<DetectBox>, ignore_class: bool) -> Vec<DetectBox> {
    if ignore_class {
        nms_with_class(iou, boxes)
    } else {
        nms_ignore_class(iou, boxes)
    }
}

pub fn nms_with_class(iou: f32, mut boxes: Vec<DetectBox>) -> Vec<DetectBox> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.

    boxes.sort_unstable_by(|a, b| match b.label.cmp(&a.label) {
        std::cmp::Ordering::Equal => b.score.total_cmp(&a.score),
        x => x,
    });

    // Outer loop over all boxes.
    for i in 0..boxes.len() {
        if boxes[i].score <= 0.0 {
            // this box was merged with a different box earlier
            continue;
        }
        for j in (i + 1)..boxes.len() {
            // Inner loop over boxes with lower score (later in the list).

            // boxes are sorted by labels first, so when we encounter a box with a different
            // class we can break
            if boxes[j].label != boxes[i].label {
                break;
            }

            if boxes[j].score <= 0.0 {
                // this box was suppressed by different box earlier
                continue;
            }

            if jaccard(&boxes[j], &boxes[i]) > iou {
                // max_box(boxes[j].bbox, &mut boxes[i].bbox);
                boxes[j].score = 0.0;
            }
        }
    }
    // Filter out boxes with a score of 0.0.
    boxes.into_iter().filter(|b| b.score > 0.0).collect()
}

pub fn nms_ignore_class(iou: f32, mut boxes: Vec<DetectBox>) -> Vec<DetectBox> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.
    boxes.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));

    // Outer loop over all boxes.
    for i in 0..boxes.len() {
        if boxes[i].score <= 0.0 {
            // this box was merged with a different box earlier
            continue;
        }
        for j in (i + 1)..boxes.len() {
            // Inner loop over boxes with lower score (later in the list).

            if boxes[j].score <= 0.0 {
                // this box was suppressed by different box earlier
                continue;
            }

            if jaccard(&boxes[j], &boxes[i]) > iou {
                // max_box(boxes[j].bbox, &mut boxes[i].bbox);
                boxes[j].score = 0.0;
            }
        }
    }
    // Filter out boxes with a score of 0.0.
    boxes.into_iter().filter(|b| b.score > 0.0).collect()
}

fn jaccard(a: &DetectBox, b: &DetectBox) -> f32 {
    let left = a.xmin.max(b.xmin);
    let top = a.ymin.max(b.ymin);
    let right = a.xmax.min(b.xmax);
    let bottom = a.ymax.min(b.ymax);

    let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
    let area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin);
    let area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin);

    // need to make sure we are not dividing by zero
    let union = (area_a + area_b - intersection).max(0.0000001);

    intersection / union
}

#[cfg(test)]
mod tests {
    use crate::{model::DetectBox, nms::jaccard};

    #[test]
    fn test_iou() {
        let iou = jaccard(
            &DetectBox {
                label: 0,
                score: 0.0,
                xmin: 0.1,
                ymin: 0.1,
                xmax: 0.2,
                ymax: 0.2,
                mask_coeff: None,
            },
            &DetectBox {
                label: 0,
                score: 0.0,
                xmin: 0.15,
                ymin: 0.15,
                xmax: 0.25,
                ymax: 0.25,
                mask_coeff: None,
            },
        );
        assert!(
            (iou - 1.0f32 / 7.0f32).abs() < 0.00001,
            "Computed IOU was not 0.57142857142"
        );
    }
    #[test]
    fn test_iou_zero() {
        let iou = jaccard(
            &DetectBox {
                label: 0,
                score: 0.0,
                xmin: 0.1,
                ymin: 0.1,
                xmax: 0.1,
                ymax: 0.2,
                mask_coeff: None,
            },
            &DetectBox {
                label: 0,
                score: 0.0,
                xmin: 0.1,
                ymin: 0.15,
                xmax: 0.1,
                ymax: 0.25,
                mask_coeff: None,
            },
        );

        assert_eq!(iou, 0.0, "Computed IOU was not 0.0");
    }
}
