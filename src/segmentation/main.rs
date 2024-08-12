// use serde::{Deserialize, Serialize};

use async_pidfd::PidFd;
use async_std::task::spawn;
use buildsegmentationmsgs::{
    build_model_info_msg, build_segmentation_msg, time_from_ns, update_model_info_msg_and_encode,
};
use cdr::{CdrLe, Infinite};
use clap::Parser;
use edgefirst_schemas::{
    self,
    edgefirst_msgs::{DmaBuf, Mask},
};
use log::{debug, error, info, trace};
use pidfd_getfd::{get_file_from_pidfd, GetFdFlags};
use setup::Settings;
use std::{
    os::fd::AsRawFd,
    path::PathBuf,
    str::FromStr,
    sync::mpsc::{self, Receiver, Sender, TryRecvError},
    time::{Duration, Instant},
};
use vaal::{self, Context};
use zenoh::{
    config::Config,
    prelude::{r#async::*, sync::SyncResolve},
    publication::Publisher,
    subscriber::FlumeSubscriber,
};
mod buildsegmentationmsgs;
mod fps;
mod setup;

#[async_std::main]
async fn main() {
    let s = Settings::parse();
    env_logger::init();
    let mut first_run = true;

    let mut config = Config::default();

    let mode = WhatAmI::from_str(&s.mode).unwrap();
    config.set_mode(Some(mode)).unwrap();
    config.connect.endpoints = s.connect.iter().map(|v| v.parse().unwrap()).collect();
    config.listen.endpoints = s.listen.iter().map(|v| v.parse().unwrap()).collect();
    let _ = config.scouting.multicast.set_enabled(Some(true));
    let _ = config
        .scouting
        .multicast
        .set_interface(Some("lo".to_string()));
    let _ = config.scouting.gossip.set_enabled(Some(true));
    let session = match zenoh::open(config.clone()).res_async().await {
        Ok(v) => v,
        Err(e) => {
            error!("Error while opening Zenoh session: {:?}", e);
            return;
        }
    }
    .into_arc();
    info!("Opened Zenoh session");

    let publ_detect = match session
        .declare_publisher(s.detect_topic.clone())
        .res_async()
        .await
    {
        Ok(v) => v,
        Err(e) => {
            error!(
                "Error while declaring detection publisher {}: {:?}",
                s.detect_topic, e
            );
            return;
        }
    };

    let publ_model_info = match session
        .declare_publisher(s.info_topic.clone())
        .res_async()
        .await
    {
        Ok(v) => v,
        Err(e) => {
            error!(
                "Error while declaring detection publisher {}: {:?}",
                s.info_topic, e
            );
            return;
        }
    };

    let sub_camera: FlumeSubscriber<'_> = session
        .declare_subscriber(&s.camera_topic)
        .res_async()
        .await
        .unwrap();
    info!("Declared subscriber on {:?}", &s.camera_topic);
    let (mask_tx, mask_rx) = mpsc::channel();
    spawn(serialize_and_publish(mask_rx, publ_detect, s.compression));
    let (heartbeat_tx, heartbeat_rx) = mpsc::channel();
    let heartbeat = spawn(heart_beat(
        sub_camera,
        mask_tx.clone(),
        publ_model_info.clone(),
        heartbeat_rx,
        s.model.clone(),
    ));

    let mut backbone = match Context::new(&s.engine) {
        Ok(v) => {
            debug!("Opened VAAL Context on {}", s.engine);
            v
        }
        Err(e) => {
            error!("Could not open VAAL Context on {}, {:?}", s.engine, e);
            return;
        }
    };
    let filename = match s.model.to_str() {
        Some(v) => v,
        None => {
            error!(
                "Cannot use file {:?}, please use only utf8 characters in
    file path",
                s.model
            );
            return;
        }
    };
    match backbone.load_model_file(filename) {
        Ok(_) => info!("Loaded backbone model {:?}", filename),
        Err(e) => {
            error!("Could not load model file {}: {:?}", filename, e);
            return;
        }
    }

    drop(heartbeat_tx);

    let mut model_info_msg =
        build_model_info_msg(time_from_ns(0u32), Some(&mut backbone), None, &s.model);
    let sub_camera = heartbeat.await;
    let timeout = Duration::from_millis(100);
    let mut fps = fps::Fps::<90>::default();
    let mut start = Instant::now();
    loop {
        let _ = sub_camera.drain();
        let mut dma_buf: DmaBuf = match sub_camera.recv_timeout(timeout) {
            Ok(v) => match cdr::deserialize(&v.payload.contiguous()) {
                Ok(v) => v,
                Err(e) => {
                    error!("Failed to deserialize message: {:?}", e);
                    continue;
                }
            },

            Err(e) => {
                error!(
                    "error receiving camera frame on {}: {:?}",
                    sub_camera.key_expr(),
                    e
                );
                continue;
            }
        };
        trace!("Recieved camera frame");
        trace!("{:?} since last inference", start.elapsed());
        start = Instant::now();
        let pidfd: PidFd = match PidFd::from_pid(dma_buf.pid as i32) {
            Ok(v) => v,
            Err(e) => {
                error!(
                    "Error getting PID {:?}, please check if the camera
    process is running: {:?}",
                    dma_buf.pid, e
                );
                continue;
            }
        };
        let fd = match get_file_from_pidfd(pidfd.as_raw_fd(), dma_buf.fd, GetFdFlags::empty()) {
            Ok(v) => v,
            Err(e) => {
                error!(
                    "Error getting Camera DMA file descriptor, please
    check if current process is running with same permissions as camera:
    {:?}",
                    e
                );
                continue;
            }
        };

        dma_buf.fd = fd.as_raw_fd();
        trace!("Opened DMA buffer from camera");
        let model_start = Instant::now();
        match run_model(&dma_buf, &backbone, fps.update()) {
            Ok(_) => {}
            Err(e) => {
                error!("Failed to run model: {:?}", e);
                return;
            }
        };
        let model_duration = model_start.elapsed();

        if first_run {
            info!(
                "Successfully recieved camera frames and run model, took {:?}",
                model_duration
            );
            first_run = false;
        }
        let _timestamp =
            dma_buf.header.stamp.nanosec as u64 + dma_buf.header.stamp.sec as u64 * 1_000_000_000;

        // let _curr_time = match clock_gettime(ClockId::CLOCK_MONOTONIC) {
        //     Ok(t) => t.num_nanoseconds() as u64,
        //     Err(e) => {
        //         error!("Could not get Monotonic clock time: {:?}", e);
        //         0
        //     }
        // };
        trace!("Model took {:?}", model_duration);
        let msg_start = Instant::now();
        let mask = build_segmentation_msg(dma_buf.header.stamp.clone(), Some(&mut backbone));
        trace!("Msg build took {:?}", msg_start.elapsed());
        let publ_start = Instant::now();
        if let Err(e) = mask_tx.send(mask) {
            error!("Publishing thread is not active: {:?}", e);
            return;
        }
        trace!(
            "Msg going to publishing thread took {:?}",
            publ_start.elapsed()
        );

        let model_info =
            update_model_info_msg_and_encode(dma_buf.header.stamp, &mut model_info_msg);
        match publ_model_info.put(model_info).res_sync() {
            Ok(_) => (),
            Err(e) => {
                error!(
                    "Error sending message on {}: {:?}",
                    publ_model_info.key_expr(),
                    e
                )
            }
        }
    }
}

#[inline(always)]
fn run_model(dma_buf: &DmaBuf, backbone: &vaal::Context, fps: f32) -> Result<(), String> {
    let start = Instant::now();
    match backbone.load_frame_dmabuf(
        None,
        dma_buf.fd,
        dma_buf.fourcc,
        dma_buf.width as i32,
        dma_buf.height as i32,
        None,
        0,
    ) {
        Err(e) => return Err(format!("Could not load frame {:?}", e)),
        Ok(_) => {
            trace!("Loaded frame into model");
        }
    };
    let load_time = start.elapsed();

    let start = Instant::now();
    if let Err(e) = backbone.run_model() {
        return Err(format!("Failed to run model: {}", e));
    }
    let model_time = start.elapsed();
    trace!(
        "Model: FPS: {:.2}, load: {:?} ms, infer: {:?} ms",
        fps, load_time, model_time,
    );
    Ok(())
}
async fn heart_beat<'a>(
    sub_camera: FlumeSubscriber<'a>,
    send_tx: Sender<Mask>,
    publ_model_info: Publisher<'_>,
    rx: Receiver<bool>,
    model_path: PathBuf,
) -> FlumeSubscriber<'a> {
    let mut model_info_msg = build_model_info_msg(time_from_ns(0u32), None, None, &model_path);
    debug!("Loading Model: {}", model_path.to_string_lossy());
    loop {
        match rx.try_recv() {
            Ok(_) => return sub_camera,
            Err(e) => match e {
                TryRecvError::Disconnected => return sub_camera,
                TryRecvError::Empty => (),
            },
        }
        let _ = sub_camera.drain();
        let mut dma_buf: DmaBuf = match sub_camera.recv_timeout(Duration::from_millis(100)) {
            Ok(v) => match cdr::deserialize(&v.payload.contiguous()) {
                Ok(v) => v,
                Err(e) => {
                    error!("Failed to deserialize message: {:?}", e);
                    continue;
                }
            },

            Err(e) => {
                error!(
                    "error receiving camera frame on {}: {:?}",
                    sub_camera.key_expr(),
                    e
                );
                continue;
            }
        };
        trace!("Recieved camera frame");

        let pidfd: PidFd = match PidFd::from_pid(dma_buf.pid as i32) {
            Ok(v) => v,
            Err(e) => {
                error!(
                    "Error getting PID {:?}, please check if the camera
process is running: {:?}",
                    dma_buf.pid, e
                );
                continue;
            }
        };
        let fd = match get_file_from_pidfd(pidfd.as_raw_fd(), dma_buf.fd, GetFdFlags::empty()) {
            Ok(v) => v,
            Err(e) => {
                error!(
                    "Error getting Camera DMA file descriptor, please
check if current process is running with same permissions as camera:
{:?}",
                    e
                );
                continue;
            }
        };
        dma_buf.fd = fd.as_raw_fd();
        trace!("Opened DMA buffer from camera");
        let mask = build_segmentation_msg(dma_buf.header.stamp.clone(), None);

        if let Err(e) = send_tx.send(mask) {
            error!("Publishing thread is not active: {:?}", e);
            return sub_camera;
        }

        let model_info =
            update_model_info_msg_and_encode(dma_buf.header.stamp.clone(), &mut model_info_msg);
        match publ_model_info.put(model_info).res_sync() {
            Ok(_) => (),
            Err(e) => {
                error!(
                    "Error sending message on {}: {:?}",
                    publ_model_info.key_expr(),
                    e
                )
            }
        }
    }
}

async fn serialize_and_publish(rx: Receiver<Mask>, publ_detect: Publisher<'_>, compression: bool) {
    loop {
        let mut msg = match rx.recv() {
            Ok(v) => v,
            Err(_) => return,
        };

        if compression {
            let compression_start = Instant::now();
            msg.mask = zstd::bulk::compress(&msg.mask, 1).unwrap();
            msg.encoding = "zstd".to_string();
            trace!("Compression takes {:?}", compression_start.elapsed());
        }

        let serialization_start = Instant::now();
        let val = Value::from(cdr::serialize::<_, _, CdrLe>(&msg, Infinite).unwrap()).encoding(
            Encoding::WithSuffix(
                KnownEncoding::AppOctetStream,
                "edgefirst_msgs/msg/Mask".into(),
            ),
        );
        trace!("Serialization takes {:?}", serialization_start.elapsed());

        let publ_start = Instant::now();
        match publ_detect.put(val).res_async().await {
            Ok(_) => trace!("Sent Detect message on {}", publ_detect.key_expr()),
            Err(e) => {
                error!(
                    "Error sending message on {}: {:?}",
                    publ_detect.key_expr(),
                    e
                )
            }
        }
        trace!("Msg sending took {:?}", publ_start.elapsed());
    }
}
