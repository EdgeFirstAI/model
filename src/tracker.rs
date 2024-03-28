use crate::setup::Settings;
use lapjv::{lapjv, Matrix};
use log::info;
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
    pub time_to_live: i32,
    pub count: i32,
    pub created: u64,
}

impl Tracklet {
    fn update(&mut self, vaalbox: &VAALBox, s: &Settings) {
        self.count += 1;
        self.time_to_live = s.track_extra_lifespan as i32;
        self.prev_boxes = *vaalbox;
    }
}

#[derive(Debug, Clone)]
pub struct TrackInfo {
    pub uuid: Uuid,
    pub count: i32,
    pub lifespan: u64,
}
const INVALID_MATCH: f32 = 10000.0;
const EPSILON: f32 = 0.00000001;
fn iou(box1: &VAALBox, box2: &VAALBox) -> f32 {
    let intersection = (box1.xmax.min(box2.xmax) - box1.xmin.max(box2.xmin))
        * (box1.ymax.min(box2.ymax) - box1.ymin.max(box2.ymin));
    let union = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin)
        + (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)
        - intersection;
    return (intersection) / (union + EPSILON);
}

fn box_cost(track_box: &VAALBox, new_box: &VAALBox, score_threshold: f32) -> f32 {
    if new_box.score < score_threshold {
        return INVALID_MATCH;
    }

    let iou = iou(track_box, new_box);
    if iou < 0.0001 {
        return INVALID_MATCH;
    }
    (1.0 - new_box.score) * (1.0 - iou) * 10000.0
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
        Matrix::from_shape_fn((dims, dims), |(x, y)| {
            if x < boxes.len() && y < self.tracklets.len() {
                if box_filter[x] || track_filter[y] {
                    INVALID_MATCH
                } else {
                    box_cost(&self.tracklets[y].prev_boxes, &boxes[x], score_threshold)
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
        boxes: &[VAALBox],
        timestamp: u64,
    ) -> Vec<Option<TrackInfo>> {
        self.frame_count += 1;
        let high_conf_ind = (0..boxes.len())
            .filter(|x| boxes[*x].score >= s.track_high_conf)
            .collect::<Vec<usize>>();
        let hc_boxes: Vec<VAALBox> = high_conf_ind.iter().map(|index| boxes[*index]).collect();
        info!("Found high conf boxes: {:?}", hc_boxes);
        let mut matched = vec![false; boxes.len()];
        let mut tracked = vec![false; self.tracklets.len()];
        let mut matched_info = vec![None; boxes.len()];
        if !self.tracklets.is_empty() {
            let costs = self.compute_costs(boxes, s.track_high_conf, &matched, &tracked);
            // With m boxes and n tracks, we compute a m x n array of costs for
            // association cost is based on 1/iou
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
                        // this is expected to happen, if there are low score boxes
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
                    self.tracklets[x].update(&boxes[i], s);
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
        info!("high conf boxes not matched: {:?}", hc_boxes);
        // try to match unmatched tracklets to low score detections as well
        if !self.tracklets.is_empty() {
            let costs = self.compute_costs(boxes, 0.0, &matched, &tracked);
            let ans = lapjv(&costs).unwrap();
            for i in 0..ans.0.len() {
                let x = ans.0[i];
                if i < boxes.len() && x < self.tracklets.len() {
                    // matched tracks
                    // We need to filter out those "invalid" assignments
                    if matched[i] || tracked[x] || costs[(i, ans.0[i])] == INVALID_MATCH {
                        // We need to filter out those "invalid" assignments
                        continue;
                    }
                    assert!(boxes[i].score < s.track_high_conf); // any boxes above the high conf threshold should either already be matched, or all tracklets are matched
                    matched[i] = true;
                    matched_info[i] = Some(TrackInfo {
                        uuid: self.tracklets[x].id,
                        count: self.tracklets[x].count,
                        lifespan: timestamp - self.tracklets[x].created,
                    });
                    assert!(!tracked[x]);
                    tracked[x] = true;
                    self.tracklets[x].update(&boxes[i], s);
                }
            }
        }

        // reduce lifespan of tracklets that didn't get matched
        for i in 0..self.tracklets.len() {
            if !tracked[i] {
                self.tracklets[i].time_to_live -= 1;
            }
        }
        // move tracklets that don't have lifespan to the removed tracklets
        for i in (0..self.tracklets.len()).rev() {
            if self.tracklets[i].time_to_live < 0 {
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
                    time_to_live: s.track_extra_lifespan as i32,
                    count: 1,
                    created: timestamp,
                });
            }
        }

        // TODO: Add some type of Kalman filter or movement prediction on the boxes

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
