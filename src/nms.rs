use std::time::Instant;

use log::debug;

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
    for i in 0..output_boxes.len().min(boxes.len()) {
        let (label, score, bbox) = boxes[i];
        output_boxes[i].xmin = bbox[0];
        output_boxes[i].ymin = bbox[1];
        output_boxes[i].xmax = bbox[2];
        output_boxes[i].ymax = bbox[3];
        output_boxes[i].score = score;
        output_boxes[i].label = label;
    }
    debug!("Box decode and nms takes {:?}", start.elapsed());
    output_boxes.len().min(boxes.len())
}

pub fn decode_boxes(
    threshold: f32,
    scores: &[f32],
    boxes: &[f32],
    num_classes: usize,
) -> Vec<(usize, f32, [f32; 4])> {
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
            out.push((label, score, [bbox[0], bbox[1], bbox[2], bbox[3]]));
        }
    }
    out
}

pub fn nms(iou: f32, mut boxes: Vec<(usize, f32, [f32; 4])>) -> Vec<(usize, f32, [f32; 4])> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.
    // let mut boxes = boxes.to_vec();
    boxes.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Outer loop over all boxes.
    for i in 0..boxes.len() {
        // Inner loop over boxes with higher score (earlier in the list).
        for j in 0..i {
            // If the boxes have the same class and the IoU is higher than the
            // threshold, the boxes are merged and the outer box is removed.
            if boxes[i].0 == boxes[j].0 && jaccard(boxes[i].2, boxes[j].2) > iou {
                let maxbox = [
                    boxes[i].2[0].min(boxes[j].2[0]),
                    boxes[i].2[1].min(boxes[j].2[1]),
                    boxes[i].2[2].max(boxes[j].2[2]),
                    boxes[i].2[3].max(boxes[j].2[3]),
                ];
                boxes[i].1 = 0.0;
                boxes[j].2 = maxbox;
            }
        }
    }
    // Filter out boxes with a score of 0.0.
    boxes
        .into_iter()
        .filter(|(_, score, _)| *score > 0.0)
        .collect()
}

fn jaccard(a: [f32; 4], b: [f32; 4]) -> f32 {
    let left = a[0].max(b[0]);
    let top = a[1].max(b[1]);
    let right = a[2].min(b[2]);
    let bottom = a[3].min(b[3]);

    let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);

    // need to make sure we are not dividing by zero
    let union = (area_a + area_b - intersection).max(0.0000001);

    intersection / union
}

#[cfg(test)]
mod tests {
    use crate::nms::{decode_boxes, jaccard};

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
    ) -> Vec<(usize, f32, [f32; 4])> {
        let scores =
            ArrayView2::from_shape([scores.len() / num_classes, num_classes], scores).unwrap();
        let boxes = ArrayView2::from_shape([boxes.len() / 4, 4], boxes).unwrap();
        Zip::from(scores.rows())
            .and(boxes.rows())
            .into_par_iter()
            .filter(|(score, _)| *score.max().unwrap() > threshold)
            .map(|(score, bbox)| {
                let label = score.argmax().unwrap();
                (label, score[label], [bbox[0], bbox[1], bbox[2], bbox[3]])
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
        let iou = jaccard([0.1, 0.1, 0.2, 0.2], [0.15, 0.15, 0.25, 0.25]);
        assert!(
            (iou - 1.0f32 / 7.0f32).abs() < 0.00001,
            "Computed IOU was not 0.57142857142"
        );
    }
    #[test]
    fn test_iou_zero() {
        let iou = jaccard([0.1, 0.1, 0.1, 0.2], [0.1, 0.15, 0.1, 0.25]);
        assert_eq!(iou, 0.0, "Computed IOU was not 0.0");
    }
}
