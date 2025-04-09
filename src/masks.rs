use crate::drain_recv;
use cdr::{CdrLe, Infinite};
use edgefirst_schemas::{self, edgefirst_msgs::Mask};
use log::{error, trace};
use std::time::Instant;
use tokio::sync::mpsc::{Receiver, Sender};
use tracing::info_span;
use zenoh::{
    bytes::{Encoding, ZBytes},
    pubsub::Publisher,
};

pub async fn mask_thread(
    mut rx: Receiver<Mask>,
    mask_classes: Vec<usize>,
    publ_mask: Publisher<'_>,
    mask_compress_tx: Option<Sender<Mask>>,
) {
    loop {
        let mut msg = match drain_recv(&mut rx).await {
            Some(v) => v,
            None => return,
        };

        let start = Instant::now();
        let mask_shape = [
            msg.height as usize,
            msg.width as usize,
            msg.mask.len() / msg.height as usize / msg.width as usize,
        ];
        if !mask_classes.is_empty() {
            let mask = info_span!("mask_slice")
                .in_scope(|| slice_mask(&msg.mask, &mask_shape, &mask_classes));
            trace!("Slice takes {:?}", start.elapsed());
            msg.mask = mask;
        }

        let (buf, enc) = info_span!("mask_publish").in_scope(|| {
            let buf = ZBytes::from(cdr::serialize::<_, _, CdrLe>(&msg, Infinite).unwrap());
            let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/Mask");
            (buf, enc)
        });
        let publ = if let Some(mask_compress_tx) = mask_compress_tx.as_ref() {
            let mask_task = publ_mask.put(buf).encoding(enc);
            let mask_send = mask_compress_tx.send(msg);
            let (publ, _send) = tokio::join!(mask_task, mask_send);
            publ
        } else {
            publ_mask.put(buf).encoding(enc).await
        };

        match publ {
            Ok(_) => trace!("Sent Mask message on {}", publ_mask.key_expr()),
            Err(e) => {
                error!("Error sending message on {}: {:?}", publ_mask.key_expr(), e)
            }
        }
    }
}

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

pub fn slice_mask(mask: &[u8], shape: &[usize; 3], classes: &[usize]) -> Vec<u8> {
    let mut new_mask = vec![0; shape[0] * shape[1] * classes.len()];
    for i in 0..shape[0] * shape[1] {
        for (ind, j) in classes.iter().enumerate() {
            if *j < shape[2] {
                new_mask[i * classes.len() + ind] = mask[i * shape[2] + j];
            }
        }
    }
    new_mask
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_slice() {
        #[rustfmt::skip]
        let mask = vec![
            0,1,2,      0,1,2,      0,1,2,      99,1,2,      0,1,2,
            0,1,2,      0,1,2,      0,1,2,      0,11,2,      17,1,2,
        ];
        let output_shape = [2, 5, 3];
        let mask_ = slice_mask(&mask, &output_shape, &[0]);
        assert_eq!(
            mask_,
            vec![0, 0, 0, 99, 0, 0, 0, 0, 0, 17,],
            "Mask is {:?} but should be {:?}",
            mask_,
            vec![0, 0, 0, 99, 0, 0, 0, 0, 0, 17,]
        );

        let mask_ = slice_mask(&mask, &output_shape, &[1]);
        assert_eq!(
            mask_,
            vec![1, 1, 1, 1, 1, 1, 1, 1, 11, 1,],
            "Mask is {:?} but should be {:?}",
            mask_,
            vec![1, 1, 1, 1, 1, 1, 1, 1, 11, 1,]
        );

        let mask_ = slice_mask(&mask, &output_shape, &[0, 2]);
        assert_eq!(
            mask_,
            vec![0, 2, 0, 2, 0, 2, 99, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 17, 2],
            "Mask is {:?} but should be {:?}",
            mask_,
            vec![0, 2, 0, 2, 0, 2, 99, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 17, 2]
        );
    }
}
