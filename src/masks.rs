// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

use crate::drain_recv;
use edgefirst_schemas::{self, edgefirst_msgs::Mask, schema_registry::SchemaType, serde_cdr};
use log::{error, trace};
use std::time::Instant;
use tokio::sync::mpsc::Receiver;
use tracing::info_span;
use zenoh::{
    bytes::{Encoding, ZBytes},
    pubsub::Publisher,
};

pub async fn mask_thread(
    mut rx: Receiver<Mask>,
    mask_classes: Vec<usize>,
    publ_mask: Publisher<'_>,
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
            let buf = ZBytes::from(serde_cdr::serialize(&msg).unwrap());
            let enc = Encoding::APPLICATION_CDR.with_schema(Mask::SCHEMA_NAME);
            (buf, enc)
        });

        match publ_mask.put(buf).encoding(enc).await {
            Ok(_) => trace!("Sent Mask message on {}", publ_mask.key_expr()),
            Err(e) => {
                error!("Error sending message on {}: {:?}", publ_mask.key_expr(), e)
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
