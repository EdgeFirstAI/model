use std::{ops::Deref, path::Path, time::Instant};

use cdr::{CdrLe, Infinite};
use deepviewrt::tensor::TensorType;
use edgefirst_schemas::{
    builtin_interfaces::Time,
    edgefirst_msgs::{model_info, Mask, ModelInfo},
    std_msgs::Header,
};
use log::{debug, error, trace};
use serde::{Deserialize, Serialize};
use vaal::Context;
use zenoh::{
    prelude::{Encoding, KnownEncoding},
    value::Value,
};

pub fn time_from_ns<T: Into<u128>>(ts: T) -> Time {
    let ts: u128 = ts.into();
    Time {
        sec: (ts / 1_000_000_000) as i32,
        nanosec: (ts % 1_000_000_000) as u32,
    }
}

fn tensor_type_to_model_info_datatype(t: TensorType) -> u8 {
    match t {
        TensorType::RAW => model_info::RAW,
        TensorType::STR => model_info::STRING,
        TensorType::I8 => model_info::INT8,
        TensorType::U8 => model_info::UINT8,
        TensorType::I16 => model_info::INT16,
        TensorType::U16 => model_info::UINT16,
        TensorType::F16 => model_info::FLOAT16,
        TensorType::I32 => model_info::INT32,
        TensorType::U32 => model_info::UINT32,
        TensorType::F32 => model_info::FLOAT32,
        TensorType::I64 => model_info::INT64,
        TensorType::U64 => model_info::UINT64,
        TensorType::F64 => model_info::FLOAT64,
    }
}

fn get_input_info(model_ctx: Option<&mut Context>) -> (Vec<u32>, u8) {
    let mut input_shape = vec![0, 0, 0, 0];
    let mut input_type = model_info::RAW;
    if let Some(ctx) = model_ctx {
        let model = match ctx.model() {
            Ok(v) => v,
            Err(_) => return (input_shape, input_type),
        };

        let inputs = match model.inputs() {
            Ok(v) => v,
            _ => return (input_shape, input_type),
        };
        let dvrt_ctx = match ctx.dvrt_context() {
            Ok(v) => v,
            Err(_) => return (input_shape, input_type),
        };
        match dvrt_ctx.tensor_index(inputs[0] as usize) {
            Ok(tensor) => {
                input_shape = tensor.shape().iter().map(|x| *x as u32).collect();
                input_type = tensor_type_to_model_info_datatype(tensor.tensor_type());
            }
            Err(_) => return (input_shape, input_type),
        };
    };
    (input_shape, input_type)
}

pub fn build_segmentation_msg(_in_time: Time, model_ctx: Option<&mut Context>) -> Mask {
    let mut output_shape: Vec<u32> = vec![0, 0, 0, 0];
    let clone_start = Instant::now();
    let mask = if let Some(model) = model_ctx {
        if let Some(tensor) = model.output_tensor(0) {
            output_shape = tensor.shape().iter().map(|x| *x as u32).collect();
            let data = tensor.mapro_u8().unwrap();
            let len = data.len();
            let mut buffer = vec![0; len];
            buffer.copy_from_slice(&(*data));
            buffer
        } else {
            error!("Did not find model output");
            Vec::new()
        }
    } else {
        Vec::new()
    };
    debug!("Clone takes {:?}", clone_start.elapsed());
    let mask_start = Instant::now();
    let msg = Mask {
        height: output_shape[1],
        width: output_shape[2],
        length: 1,
        encoding: "".to_string(),
        mask,
    };
    debug!("Making mask struct takes {:?}", mask_start.elapsed());
    // let serialization_start = Instant::now();
    // let val = Value::from(cdr::serialize::<_, _, CdrLe>(&msg,
    // Infinite).unwrap()).encoding(     Encoding::WithSuffix(
    //         KnownEncoding::AppOctetStream,
    //         "edgefirst_msgs/msg/Mask".into(),
    //     ),
    // );
    // debug!("Serialization takes {:?}", serialization_start.elapsed());
    return msg;
}

pub fn build_model_info_msg(
    in_time: Time,
    model_ctx: Option<&mut Context>,
    decoder_ctx: Option<&mut Context>,
    path: &Path,
) -> ModelInfo {
    let mut output_shape: Vec<u32> = vec![0, 0, 0, 0];
    let mut output_type = model_info::RAW;
    let output_ctx = match decoder_ctx {
        Some(_) => &decoder_ctx,
        None => &model_ctx,
    };
    if let Some(ctx) = output_ctx {
        if let Some(tensor) = ctx.output_tensor(0) {
            output_shape = tensor.shape().iter().map(|x| *x as u32).collect();
            output_type = tensor_type_to_model_info_datatype(tensor.tensor_type());
        }
    };
    let labels = match output_ctx {
        Some(ref ctx) => ctx.labels().into_iter().map(String::from).collect(),
        None => Vec::new(),
    };

    let model_format = match path.extension() {
        // , HailoRT, RKNN, TensorRT, TFLite
        Some(v) => match v.to_string_lossy().to_ascii_lowercase().as_str() {
            "rtm" => String::from("DeepViewRT"),
            "rknn" => String::from("RKNN"),
            "tflite" => String::from("TFLite"),
            "hef" => String::from("HailoRT"),
            _ => v.to_string_lossy().into_owned(),
        },
        None => String::from("unknown"),
    };

    let model_name = match model_ctx {
        Some(ref ctx) if ctx.model().is_err() => String::from("No Model"),
        Some(ref ctx)
            if ctx.model().unwrap().name().is_err()
                || ctx.model().unwrap().name().unwrap().is_empty() =>
        {
            //the path cannot end with a `..` otherwise the model would not have loaded
            path.file_name().unwrap().to_string_lossy().into_owned()
        }
        Some(ref ctx) => ctx.model().unwrap().name().unwrap().to_owned(),
        None => String::from("Loading Model..."),
    };
    debug!("Model name = {}", model_name);
    let model_type = String::from("Detection");

    let (input_shape, input_type) = get_input_info(model_ctx);

    ModelInfo {
        header: Header {
            stamp: in_time.clone(),
            frame_id: String::new(),
        },
        labels, // input_shape = model_ctx.
        input_shape,
        input_type,
        output_shape,
        output_type,
        model_format,
        model_name,
        model_type,
    }
}

//Updated the ModelInfo message with the given timestamp, and then encodes it
// into a Zenoh Value
pub fn update_model_info_msg_and_encode(timestamp: Time, msg: &mut ModelInfo) -> Value {
    msg.header.stamp = timestamp;
    Value::from(cdr::serialize::<_, _, CdrLe>(&msg, Infinite).unwrap()).encoding(
        Encoding::WithSuffix(
            KnownEncoding::AppOctetStream,
            "edgefirst_msgs/msg/ModelInfo".into(),
        ),
    )
}
