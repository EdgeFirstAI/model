// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

use crate::drain_recv;
use edgefirst_schemas::{self, edgefirst_msgs::Mask, schema_registry::SchemaType, serde_cdr};
use log::{error, trace};
use tokio::sync::mpsc::Receiver;
use zenoh::{
    bytes::{Encoding, ZBytes},
    pubsub::Publisher,
};

pub async fn mask_thread(mut rx: Receiver<Mask>, publ_mask: Publisher<'_>) {
    loop {
        let msg = match drain_recv(&mut rx).await {
            Some(v) => v,
            None => return,
        };

        let buf = ZBytes::from(serde_cdr::serialize(&msg).unwrap());
        let enc = Encoding::APPLICATION_CDR.with_schema(Mask::SCHEMA_NAME);

        match publ_mask.put(buf).encoding(enc).await {
            Ok(_) => trace!("Sent Mask message on {}", publ_mask.key_expr()),
            Err(e) => {
                error!("Error sending message on {}: {:?}", publ_mask.key_expr(), e)
            }
        }
    }
}
