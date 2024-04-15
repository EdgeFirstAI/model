use zenoh_ros_type::{
    builtin_interfaces::Time,
    point_annotation_type::{LINE_LOOP, UNKNOWN},
    std_msgs::Header,
    DetectBox2D, DetectBoxes2D, DetectTrack, FoxgloveColor, FoxgloveImageAnnotations,
    FoxglovePoint2, FoxglovePointAnnotations, FoxgloveTextAnnotations,
};

use crate::{Box2D, LabelSetting};

const WHITE: FoxgloveColor = FoxgloveColor {
    r: 1.0,
    g: 1.0,
    b: 1.0,
    a: 1.0,
};
const TRANSPARENT: FoxgloveColor = FoxgloveColor {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 0.0,
};

fn u128_to_foxglove_color(hexcode: u128) -> FoxgloveColor {
    const BYTES_PER_CHANNEL: u8 = 8;
    const FACTOR: u32 = (1 << BYTES_PER_CHANNEL) - 1;

    // only use the first 32 bits
    let hexcode = (hexcode >> (128 - (4 * BYTES_PER_CHANNEL))) as u32;
    FoxgloveColor {
        r: ((hexcode >> (BYTES_PER_CHANNEL * 3)) & FACTOR) as f64 / FACTOR as f64,
        g: ((hexcode >> (BYTES_PER_CHANNEL * 2)) & FACTOR) as f64 / FACTOR as f64,
        b: ((hexcode >> BYTES_PER_CHANNEL) & FACTOR) as f64 / FACTOR as f64,
        a: 1.0,
    }
}
pub fn build_image_annotations_msg(
    boxes: &[Box2D],
    timestamp: Time,
    stream_width: f64,
    stream_height: f64,
    msg: &str,
    labels: LabelSetting,
) -> FoxgloveImageAnnotations {
    let mut annotations = FoxgloveImageAnnotations {
        circles: Vec::new(),
        points: Vec::new(),
        texts: Vec::new(),
    };

    let empty_points = FoxglovePointAnnotations {
        timestamp: timestamp.clone(),
        type_: UNKNOWN,
        points: Vec::new(),
        outline_color: WHITE.clone(),
        outline_colors: Vec::new(),
        fill_color: TRANSPARENT.clone(),
        thickness: 2.0,
    };

    let empty_text = FoxgloveTextAnnotations {
        timestamp: timestamp.clone(),
        text: msg.to_owned(),
        position: FoxglovePoint2 {
            x: stream_width * 0.025,
            y: stream_height * 0.95,
        },
        font_size: 0.015 * stream_width.max(stream_height),
        text_color: WHITE.clone(),
        background_color: TRANSPARENT.clone(),
    };

    annotations.points.push(empty_points);
    annotations.texts.push(empty_text);

    for b in boxes.iter() {
        let color = match &b.track {
            None => WHITE.clone(),
            Some(track) => u128_to_foxglove_color(track.uuid.as_u128()),
        };
        let outline_colors = vec![color.clone(), color.clone(), color.clone(), color.clone()];
        let points = vec![
            FoxglovePoint2 {
                x: b.xmin * stream_width,
                y: b.ymin * stream_height,
            },
            FoxglovePoint2 {
                x: b.xmax * stream_width,
                y: b.ymin * stream_height,
            },
            FoxglovePoint2 {
                x: b.xmax * stream_width,
                y: b.ymax * stream_height,
            },
            FoxglovePoint2 {
                x: b.xmin * stream_width,
                y: b.ymax * stream_height,
            },
        ];
        let points = FoxglovePointAnnotations {
            timestamp: timestamp.clone(),
            type_: LINE_LOOP,
            points,
            outline_color: color.clone(),
            outline_colors,
            fill_color: TRANSPARENT.clone(),
            thickness: 2.0,
        };

        match labels {
            LabelSetting::Index => format!("{:.2}", b.index),
            LabelSetting::Score => format!("{:.2}", b.score),
            LabelSetting::Label => b.label.clone(),
            LabelSetting::LabelScore => {
                format!("{} {:.2}", b.label, b.score)
            }
            LabelSetting::Track => match &b.track {
                None => format!("{:.2}", b.score),
                // only shows first 8 characters of the UUID
                Some(v) => format!("{}", v.uuid.to_string().split_at(8).0),
            },
        };

        let text = FoxgloveTextAnnotations {
            timestamp: timestamp.clone(),
            text: b.label.clone(),
            position: FoxglovePoint2 {
                x: b.xmin * stream_width,
                y: b.ymin * stream_height,
            },
            font_size: 0.02 * stream_width.max(stream_height),
            text_color: color.clone(),
            background_color: TRANSPARENT.clone(),
        };
        annotations.points.push(points);
        annotations.texts.push(text);
    }
    annotations
}

pub fn time_from_ns(ts: u64) -> Time {
    Time {
        sec: (ts / 1000_000_000) as i32,
        nanosec: (ts % 1000_000_000) as u32,
    }
}

impl From<&Box2D> for DetectBox2D {
    fn from(box2d: &Box2D) -> Self {
        let track = match &box2d.track {
            Some(v) => DetectTrack {
                id: v.uuid.to_string(),
                lifetime: v.count,
                created: time_from_ns(v.created),
            },
            None => DetectTrack {
                id: String::new(),
                lifetime: 0,
                created: time_from_ns(0),
            },
        };
        DetectBox2D {
            center_x: (box2d.xmax + box2d.xmin) as f32,
            center_y: (box2d.ymax + box2d.ymin) as f32,
            width: (box2d.xmax - box2d.xmin) as f32,
            height: (box2d.ymax - box2d.ymin) as f32,
            label: box2d.label.clone(),
            score: box2d.score as f32,
            distance: 0.0,
            speed: 0.0,
            is_tracked: box2d.track.is_some(),
            track,
        }
    }
}

pub fn build_detectboxes2d_msg(
    boxes: &[Box2D],
    in_time: Time,
    model_time: Time,
    curr_time: Time,
) -> DetectBoxes2D {
    let mut new_boxes = Vec::new();
    for b in boxes {
        new_boxes.push(b.into());
    }

    DetectBoxes2D {
        header: Header {
            stamp: in_time.clone(),
            frame_id: String::new(),
        },
        input_timestamp: in_time,
        model_time,
        output_time: curr_time,
        boxes: new_boxes,
    }
}
