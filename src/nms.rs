use std::time::Instant;

use log::trace;

use crate::model::DetectBox;

pub fn decode_boxes_and_nms(
    score_threshold: f32,
    iou_threshold: f32,
    scores_tensor: &[f32],
    boxes_tensor: &[f32],
    num_classes: usize,
    output_boxes: &mut [DetectBox],
) -> usize {
    let start = Instant::now();
    let boxes = decode_boxes(score_threshold, scores_tensor, boxes_tensor, num_classes);
    let boxes = nms(iou_threshold, boxes);
    let len = output_boxes.len().min(boxes.len());
    for (out, b) in output_boxes.iter_mut().zip(boxes) {
        *out = b;
    }
    trace!("Box decode and nms takes {:?}", start.elapsed());
    len
}

pub fn decode_boxes(
    threshold: f32,
    scores: &[f32],
    boxes: &[f32],
    num_classes: usize,
) -> Vec<DetectBox> {
    assert_eq!(scores.len() / num_classes, boxes.len() / 4);
    let box_count = scores.len() / num_classes;
    let mut out = Vec::new();
    for i in 0..box_count {
        let mut score = 0.0;
        let mut label = 0;
        for j in 0..num_classes {
            if scores[i * num_classes + j] > score {
                score = scores[i * num_classes + j];
                label = j;
            }
        }
        let bbox = &boxes[i * 4..(i + 1) * 4];
        if score > threshold {
            out.push(DetectBox {
                xmin: bbox[0],
                ymin: bbox[1],
                xmax: bbox[2],
                ymax: bbox[3],
                score,
                label,
            });
        }
    }
    out
}

pub fn nms(iou: f32, mut boxes: Vec<DetectBox>) -> Vec<DetectBox> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.
    boxes.sort_unstable_by(|a, b| (b.label, b.score).partial_cmp(&(a.label, a.score)).unwrap());

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
    use crate::{
        model::DetectBox,
        nms::{decode_boxes, jaccard},
    };

    use ndarray::{
        ArrayView2, Zip,
        parallel::prelude::{IntoParallelIterator, ParallelIterator},
    };
    use ndarray_stats::QuantileExt as _;
    use rand::random;
    pub fn decode_boxes1(
        threshold: f32,
        scores: &[f32],
        boxes: &[f32],
        num_classes: usize,
    ) -> Vec<DetectBox> {
        let scores =
            ArrayView2::from_shape([scores.len() / num_classes, num_classes], scores).unwrap();
        let boxes = ArrayView2::from_shape([boxes.len() / 4, 4], boxes).unwrap();
        Zip::from(scores.rows())
            .and(boxes.rows())
            .into_par_iter()
            .filter(|(score, _)| *score.max().unwrap() > threshold)
            .map(|(score, bbox)| {
                let label = score.argmax().unwrap();
                DetectBox {
                    xmin: bbox[0],
                    ymin: bbox[1],
                    xmax: bbox[2],
                    ymax: bbox[3],
                    score: score[label],
                    label,
                }
            })
            .collect()
    }

    #[test]
    fn box_decoding() {
        const NUM_BOXES: usize = 6009;
        const NUM_CLASSES: usize = 10;
        let box_data: [f32; NUM_BOXES * 4] = random();
        let score_data: [f32; NUM_BOXES * NUM_CLASSES] = random();
        let decoded_boxes_ndarray = decode_boxes1(0.5, &score_data, &box_data, NUM_CLASSES);
        let decoded_boxes = decode_boxes(0.5, &score_data, &box_data, NUM_CLASSES);
        assert_eq!(
            decoded_boxes_ndarray, decoded_boxes,
            "Decoded boxes were not equal"
        );
    }

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
            },
            &DetectBox {
                label: 0,
                score: 0.0,
                xmin: 0.15,
                ymin: 0.15,
                xmax: 0.25,
                ymax: 0.25,
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
            },
            &DetectBox {
                label: 0,
                score: 0.0,
                xmin: 0.1,
                ymin: 0.15,
                xmax: 0.1,
                ymax: 0.25,
            },
        );

        assert_eq!(iou, 0.0, "Computed IOU was not 0.0");
    }
}
