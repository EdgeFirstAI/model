use crate::drain_recv;
use cdr::{CdrLe, Infinite};
use edgefirst_schemas::{self, edgefirst_msgs::Mask};
use log::{error, trace};
use std::time::Instant;
use tokio::sync::mpsc::Receiver;
use tracing::info_span;
use zenoh::{
    bytes::{Encoding, ZBytes},
    pubsub::Publisher,
};

pub async fn mask_compress_thread(
    mut rx: Receiver<Mask>,
    publ_mask_compressed: Publisher<'_>,
    level: i32,
) {
    loop {
        let mut msg = match drain_recv(&mut rx).await {
            Some(v) => v,
            None => return,
        };

        let (buf, enc) = info_span!("mask_compressed_publish").in_scope(|| {
            let start = Instant::now();
            let start_size = msg.mask.len();
            msg.mask = zstd::bulk::compress(&msg.mask, level).unwrap();
            trace!(
                "compression takes {:?} with ratio {:3}%",
                start.elapsed(),
                msg.mask.len() as f64 / start_size as f64 * 100.0
            );
            msg.encoding = "zstd".to_string();

            let buf = ZBytes::from(cdr::serialize::<_, _, CdrLe>(&msg, Infinite).unwrap());
            let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/Mask");

            (buf, enc)
        });
        let publ = publ_mask_compressed.put(buf).encoding(enc).await;
        match publ {
            Ok(_) => trace!(
                "Sent compressed Mask message on {}",
                publ_mask_compressed.key_expr()
            ),
            Err(e) => {
                error!(
                    "Error sending message on {}: {:?}",
                    publ_mask_compressed.key_expr(),
                    e
                )
            }
        }
    }
}
