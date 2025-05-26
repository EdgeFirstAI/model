use crate::{
    metadata_schema_generated::tflite::root_as_model_metadata,
    schema_generated::tflite::root_as_model,
};

#[derive(Debug, Default, Eq, PartialEq)]
pub struct Metadata {
    pub name: Option<String>,
    pub version: Option<String>,
    pub description: Option<String>,
    pub author: Option<String>,
    pub license: Option<String>,
    pub min_parser_version: Option<String>,
}

const kMetadataBufferName: &str = "TFLITE_METADATA";
pub fn get_model_metadata(model: &[u8]) -> Metadata {
    let mut metadata = Metadata::default();
    let m = match root_as_model(model) {
        Ok(v) => v,
        Err(_) => return metadata,
    };
    let model_desc = m.description();
    for i in m.metadata().unwrap() {
        if i.name().is_none_or(|n| n != kMetadataBufferName) {
            continue;
        }

        let buffer_index = i.buffer();
        let buffers = match m.buffers() {
            Some(v) => v,
            None => return metadata,
        };
        let metadata_buffer = match buffers.get(buffer_index as usize).data() {
            Some(v) => v,
            None => return metadata,
        };
        let model_metadata = match root_as_model_metadata(metadata_buffer.bytes()) {
            Ok(v) => v,
            Err(_) => return metadata,
        };
        metadata.name = model_metadata.name().map(|x| x.to_owned());
        metadata.description = match (model_desc, model_metadata.description()) {
            (Some(s1), Some(s2)) => Some(format!("{s1} {s2}")),
            (Some(s1), None) => Some(s1.to_owned()),
            (None, Some(s2)) => Some(s2.to_owned()),
            (None, None) => None,
        };

        metadata.author = model_metadata.author().map(|x| x.to_owned());
        metadata.license = model_metadata.license().map(|x| x.to_owned());
        metadata.min_parser_version = model_metadata.min_parser_version().map(|x| x.to_owned());
        metadata.version = model_metadata.version().map(|x| x.to_owned());
    }
    metadata
}
