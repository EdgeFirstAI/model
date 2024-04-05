use crate::{
    kalman::{self, ConstantVelocityXYAHModel2, GatingDistanceMetric},
    setup::Settings,
};
use lapjv::{lapjv, Matrix};
use log::{debug, info};
use nalgebra::{
    distance, Dyn, Matrix4, OMatrix, OVector, SVector, Vector2, Vector4, VectorN, U4, U8,
};
use uuid::Uuid;
use vaal::VAALBox;

pub struct ByteTrack {
    // tracklets;
    pub tracklets: Vec<Tracklet>,
    pub lost_tracks: Vec<Tracklet>,
    pub removed_tracks: Vec<Tracklet>,
    pub frame_count: i32,
}
#[derive(Debug, Clone)]
pub struct Tracklet {
    pub id: Uuid,
    pub prev_boxes: vaal::VAALBox,
    pub filter: ConstantVelocityXYAHModel2<f32>,
    pub time_to_live: i32,
    pub count: i32,
    pub created: u64,
}

impl Tracklet {
    fn update(&mut self, vaalbox: &VAALBox, s: &Settings) {
        self.count += 1;
        self.time_to_live = s.track_extra_lifespan as i32;
        self.prev_boxes = *vaalbox;
        self.filter.update(&vaalbox_to_xyah(vaalbox));
    }
}

fn vaalbox_to_xyah(vaal_box: &VAALBox) -> [f32; 4] {
    let x = (vaal_box.xmax + vaal_box.xmin) / 2.0;
    let y = (vaal_box.ymax + vaal_box.ymin) / 2.0;
    let a = (vaal_box.xmax - vaal_box.xmin) / (vaal_box.ymax - vaal_box.ymin);
    let h = vaal_box.ymax - vaal_box.ymin;
    return [x, y, a, h];
}
#[derive(Debug, Clone)]
pub struct TrackInfo {
    pub uuid: Uuid,
    pub count: i32,
    pub lifespan: u64,
}
const INVALID_MATCH: f32 = 1000000.0;
const EPSILON: f32 = 0.0000001;
fn iou(box1: &VAALBox, box2: &VAALBox) -> f32 {
    let intersection = (box1.xmax.min(box2.xmax) - box1.xmin.max(box2.xmin)).max(0.0)
        * (box1.ymax.min(box2.ymax) - box1.ymin.max(box2.ymin)).max(0.0);

    if intersection <= EPSILON {
        return 0.0;
    }

    let union = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin)
        + (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)
        - intersection;

    if union <= 0.0 {
        return 0.0;
    }
    assert!(intersection <= union, "box1: {:#?} box2: {:#?}", box1, box2);
    return intersection / union;
}

fn box_cost(track: &Tracklet, new_box: &VAALBox, distance: f32, score_threshold: f32) -> f32 {
    if new_box.score < score_threshold {
        return INVALID_MATCH;
    }
    // (1.5 - new_box.score * new_box.score) * (distance)

    // use iou between predicted box and real box:
    let predicted_xyah = track.filter.mean.as_slice();
    let x_ = predicted_xyah[0];
    let y_ = predicted_xyah[1];
    let a_ = predicted_xyah[2];
    let h_ = predicted_xyah[3];
    let w_ = h_ * a_;
    let expected = VAALBox {
        xmin: x_ - w_ / 2.0,
        xmax: x_ + w_ / 2.0,
        ymin: y_ - h_ / 2.0,
        ymax: y_ + h_ / 2.0,
        score: 0.0,
        label: 0,
    };
    let iou = iou(&expected, new_box);
    if iou <= 0.25 {
        return INVALID_MATCH;
    }
    let cost = (1.5 - new_box.score) + (1.5 - iou);
    assert!(cost >= 0.0, "score = {}, iou = {}", new_box.score, iou);

    // if distance > 0.1 {
    //     return INVALID_MATCH;
    // }
    // let cost = (1.5 - new_box.score) * distance;

    cost
}

impl ByteTrack {
    pub fn new() -> ByteTrack {
        ByteTrack {
            tracklets: vec![],
            lost_tracks: vec![],
            removed_tracks: vec![],
            frame_count: 0,
        }
    }

    // fn getMultFromScore(score: f32) {

    // }

    fn compute_costs(
        &mut self,
        boxes: &[VAALBox],
        score_threshold: f32,
        box_filter: &[bool],
        track_filter: &[bool],
    ) -> Matrix<f32> {
        // costs matrix must be square
        let dims = boxes.len().max(self.tracklets.len());
        let mut measurements = OMatrix::<f32, Dyn, U4>::from_element(boxes.len(), 0.0);
        for (i, mut row) in measurements.row_iter_mut().enumerate() {
            row.copy_from_slice(&vaalbox_to_xyah(&boxes[i]));
        }

        // TODO: use matrix math for IOU, should speed up computation, and store it in
        // distances

        // distances range from 0 to 1e6 for maha
        // let mut distances =
        //     OMatrix::<f32, Dyn, Dyn>::from_element(boxes.len(), self.tracklets.len(),
        // 0.0);
        // for (i, mut column) in distances.column_iter_mut().enumerate() {
        //     let dist = self.tracklets[i].filter.gating_distance(
        //         &measurements,
        //         false,
        //         GatingDistanceMetric::Gaussian,
        //     );
        //     column.copy_from(&dist);
        // }

        Matrix::from_shape_fn((dims, dims), |(x, y)| {
            if x < boxes.len() && y < self.tracklets.len() {
                if box_filter[x] || track_filter[y] {
                    INVALID_MATCH
                } else {
                    box_cost(
                        &self.tracklets[y],
                        &boxes[x],
                        // distances[(x, y)],
                        0.0,
                        score_threshold,
                    )
                }
            } else {
                0.0
            }
        })
    }

    // pub fn default() -> ByteTrack {}

    pub fn update(
        &mut self,
        s: &Settings,
        boxes: &mut [VAALBox],
        timestamp: u64,
    ) -> Vec<Option<TrackInfo>> {
        self.frame_count += 1;
        let high_conf_ind = (0..boxes.len())
            .filter(|x| boxes[*x].score >= s.track_high_conf)
            .collect::<Vec<usize>>();
        let hc_boxes: Vec<VAALBox> = high_conf_ind.iter().map(|index| boxes[*index]).collect();
        debug!("Found high conf boxes: {:?}", hc_boxes);
        let mut matched = vec![false; boxes.len()];
        let mut tracked = vec![false; self.tracklets.len()];
        let mut matched_info = vec![None; boxes.len()];
        if !self.tracklets.is_empty() {
            for track in &mut self.tracklets {
                track.filter.predict();
            }
            let costs = self.compute_costs(boxes, s.track_high_conf, &matched, &tracked);
            // With m boxes and n tracks, we compute a m x n array of costs for
            // association cost is based on distance computed by the Kalman Filter
            // Then we use lapjv (linear assignment) to minimize the cost of
            // matching tracks to boxes
            // The linear assignment will still assign some tracks to out of threshold
            // scores/filtered tracks/filtered boxes But it will try to minimize
            // the number of "invalid" assignments, since those are just very high costs
            let ans = lapjv(&costs).unwrap();
            for i in 0..ans.0.len() {
                let x = ans.0[i];
                if i < boxes.len() && x < self.tracklets.len() {
                    // We need to filter out those "invalid" assignments
                    if costs[(i, ans.0[i])] >= INVALID_MATCH {
                        continue;
                    }
                    matched[i] = true;
                    matched_info[i] = Some(TrackInfo {
                        uuid: self.tracklets[x].id,
                        count: self.tracklets[x].count,
                        lifespan: timestamp - self.tracklets[x].created,
                    });
                    assert!(!tracked[x]);
                    tracked[x] = true;
                    let predicted_xyah = self.tracklets[x].filter.mean.as_slice();
                    let x_ = predicted_xyah[0];
                    let y_ = predicted_xyah[1];
                    let a_ = predicted_xyah[2];
                    let h_ = predicted_xyah[3];

                    self.tracklets[x].update(&boxes[i], s);

                    let w_ = h_ * a_;
                    boxes[i].xmin = x_ - w_ / 2.0;
                    boxes[i].xmax = x_ + w_ / 2.0;
                    boxes[i].ymin = y_ - h_ / 2.0;
                    boxes[i].ymax = y_ + h_ / 2.0;
                }
            }
        }
        let match_high_conf_ind = (0..boxes.len())
            .filter(|x| high_conf_ind.contains(x) && !matched[*x])
            .collect::<Vec<usize>>();
        let hc_boxes: Vec<VAALBox> = match_high_conf_ind
            .iter()
            .map(|index| boxes[*index])
            .collect();
        debug!("high conf boxes not matched: {:?}", hc_boxes);
        // try to match unmatched tracklets to low score detections as well
        if !self.tracklets.is_empty() {
            let costs = self.compute_costs(boxes, 0.0, &matched, &tracked);
            let ans = lapjv(&costs).unwrap();
            for i in 0..ans.0.len() {
                let x = ans.0[i];
                if i < boxes.len() && x < self.tracklets.len() {
                    // matched tracks
                    // We need to filter out those "invalid" assignments
                    if matched[i] || tracked[x] || (costs[(i, x)] >= INVALID_MATCH) {
                        // We need to filter out those "invalid" assignments
                        continue;
                    }
                    matched[i] = true;
                    matched_info[i] = Some(TrackInfo {
                        uuid: self.tracklets[x].id,
                        count: self.tracklets[x].count,
                        lifespan: timestamp - self.tracklets[x].created,
                    });
                    info!(
                        "Cost: {} Box: {:#?} UUID: {} Mean: {}",
                        costs[(i, x)],
                        boxes[i],
                        self.tracklets[x].id,
                        self.tracklets[x].filter.mean
                    );
                    assert!(!tracked[x]);
                    tracked[x] = true;
                    let predicted_xyah = self.tracklets[x].filter.mean.as_slice();
                    let x_ = predicted_xyah[0];
                    let y_ = predicted_xyah[1];
                    let a_ = predicted_xyah[2];
                    let h_ = predicted_xyah[3];

                    self.tracklets[x].update(&boxes[i], s);

                    let w_ = h_ * a_;
                    boxes[i].xmin = x_ - w_ / 2.0;
                    boxes[i].xmax = x_ + w_ / 2.0;
                    boxes[i].ymin = y_ - h_ / 2.0;
                    boxes[i].ymax = y_ + h_ / 2.0;
                }
            }
        }

        // reduce lifespan of tracklets that didn't get matched
        for i in 0..self.tracklets.len() {
            if !tracked[i] {
                info!("Tracklet without match: {:#?}", self.tracklets[i]);
                self.tracklets[i].time_to_live -= 1;
            }
        }
        // move tracklets that don't have lifespan to the removed tracklets
        for i in (0..self.tracklets.len()).rev() {
            if self.tracklets[i].time_to_live < 0 {
                info!("Tracklet removed: {:?}", self.tracklets[i].id);
                let _ = self.tracklets.swap_remove(i);
            }
        }

        // unmatched high score boxes are then used to make new tracks
        for i in high_conf_ind {
            if !matched[i] {
                let id = Uuid::new_v4();
                matched_info[i] = Some(TrackInfo {
                    uuid: id,
                    count: 1,
                    lifespan: 0,
                });
                self.tracklets.push(Tracklet {
                    id,
                    prev_boxes: boxes[i],
                    filter: ConstantVelocityXYAHModel2::new(&vaalbox_to_xyah(&boxes[i])),
                    time_to_live: s.track_extra_lifespan as i32,
                    count: 1,
                    created: timestamp,
                });
            }
        }
        matched_info
    }
}

// #[cfg(test)]
// mod tests {
//     use vaal::VAALBox;

//     use super::ByteTrack;

//     #[test]
//     fn bytetrack() {
//         let mut bt = ByteTrack::new();
//         let mut boxes = vec![VAALBox {
//             label: 0,
//             xmin: 0.5,
//             ymin: 0.5,
//             xmax: 0.6,
//             ymax: 0.6,
//             score: 0.6,
//         }];
//         bt.update(&mut boxes);

//         boxes[0].score = 0.4;
//         boxes.push(VAALBox {
//             label: 0,
//             xmin: 0.3,
//             ymin: 0.3,
//             xmax: 0.4,
//             ymax: 0.4,
//             score: 0.6,
//         });

//         // boxes.reverse();
//         bt.update(&mut boxes);
//     }
// }
