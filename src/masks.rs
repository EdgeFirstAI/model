// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

use crate::{drain_recv, zenoh_timestamp};
use edgefirst_schemas::{builtin_interfaces::Time, edgefirst_msgs::Mask};
use log::{error, trace};
use tokio::sync::mpsc::Receiver;
use zenoh::{
    bytes::{Encoding, ZBytes},
    pubsub::Publisher,
};

/// Publishes masks paired with the camera frame stamp they were computed
/// from. `Mask` has no header, so the Zenoh sample timestamp is the only
/// record of that stamp.
pub async fn mask_thread(
    mut rx: Receiver<(Time, Mask<Vec<u8>>)>,
    publ_mask: Publisher<'_>,
    session: zenoh::Session,
) {
    loop {
        let (stamp, msg) = match drain_recv(&mut rx).await {
            Some(v) => v,
            None => return,
        };

        let buf = ZBytes::from(msg.into_cdr());
        let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/Mask");

        match publ_mask
            .put(buf)
            .encoding(enc)
            .timestamp(zenoh_timestamp(&session, stamp))
            .await
        {
            Ok(_) => trace!("Sent Mask message on {}", publ_mask.key_expr()),
            Err(e) => {
                error!("Error sending message on {}: {:?}", publ_mask.key_expr(), e)
            }
        }
    }
}
