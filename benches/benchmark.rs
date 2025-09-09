use edgefirst_model::{
    model::{
        DataType, Decoder, DetectBox, Detection, decode_detection_outputs, fast_sigmoid,
        fast_sigmoid2, sigmoid,
    },
    nms::{decode_boxes, decode_boxes_and_nms, nms},
};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::{Layer, Registry, layer::SubscriberExt};

#[divan::bench]
fn sigmoid_bench(bencher: divan::Bencher) {
    let output1 = include_str!("benchmark_data/output_34x60x30.txt")
        .lines()
        .flat_map(|x| x.split_whitespace().map(|y| y.parse::<f32>().unwrap()))
        .collect::<Vec<_>>();

    let output2 = include_str!("benchmark_data/output_68x120x30.txt")
        .lines()
        .flat_map(|x| x.split_whitespace().map(|y| y.parse::<f32>().unwrap()))
        .collect::<Vec<_>>();
    bencher
        .with_inputs(|| (output1.clone(), output2.clone()))
        .bench_local_values(|(mut num1, mut num2)| {
            num1.chunks_exact_mut(4)
                .for_each(|x| x.iter_mut().for_each(|x| *x = sigmoid(*x)));
            num2.chunks_exact_mut(4)
                .for_each(|x| x.iter_mut().for_each(|x| *x = sigmoid(*x)));
            // num1.iter_mut().for_each(|x| *x = sigmoid(*x));
            // num2.iter_mut().for_each(|x| *x = sigmoid(*x));
        });
}

#[divan::bench]
fn sigmoid_fast_bench(bencher: divan::Bencher) {
    let output1 = include_str!("benchmark_data/output_34x60x30.txt")
        .lines()
        .flat_map(|x| x.split_whitespace().map(|y| y.parse::<f32>().unwrap()))
        .collect::<Vec<_>>();

    let output2 = include_str!("benchmark_data/output_68x120x30.txt")
        .lines()
        .flat_map(|x| x.split_whitespace().map(|y| y.parse::<f32>().unwrap()))
        .collect::<Vec<_>>();
    bencher
        .with_inputs(|| (output1.clone(), output2.clone()))
        .bench_local_values(|(mut num1, mut num2)| {
            num1.chunks_exact_mut(4)
                .for_each(|x| x.iter_mut().for_each(|x| *x = fast_sigmoid(*x)));
            let start = num1.len() / 4 * 4;
            num1[start..].iter_mut().for_each(|x| *x = fast_sigmoid(*x));

            num2.chunks_exact_mut(4)
                .for_each(|x| x.iter_mut().for_each(|x| *x = fast_sigmoid(*x)));
            let start = num2.len() / 4 * 4;
            num2[start..].iter_mut().for_each(|x| *x = fast_sigmoid(*x));
        });
}

#[divan::bench]
fn sigmoid_fast_bench2(bencher: divan::Bencher) {
    let output1 = include_str!("benchmark_data/output_34x60x30.txt")
        .lines()
        .flat_map(|x| x.split_whitespace().map(|y| y.parse::<f32>().unwrap()))
        .collect::<Vec<_>>();

    let output2 = include_str!("benchmark_data/output_68x120x30.txt")
        .lines()
        .flat_map(|x| x.split_whitespace().map(|y| y.parse::<f32>().unwrap()))
        .collect::<Vec<_>>();
    bencher
        .with_inputs(|| (output1.clone(), output2.clone()))
        .bench_local_values(|(mut num1, mut num2)| {
            // this simd version is slower than the normal
            // num1.chunks_exact_mut(4)
            //     .for_each(|x| x.iter_mut().for_each(|x| *x = fast_sigmoid2_(*x)));
            // num2.chunks_exact_mut(4)
            //     .for_each(|x| x.iter_mut().for_each(|x| *x = fast_sigmoid2_(*x)));

            num1.iter_mut().for_each(|x| *x = fast_sigmoid2(*x));
            num2.iter_mut().for_each(|x| *x = fast_sigmoid2(*x));
        });
}

#[divan::bench]
fn full_decode_benchmark(bencher: divan::Bencher) {
    let output1 = include_str!("benchmark_data/output_34x60x30.txt")
        .lines()
        .flat_map(|x| x.split_whitespace().map(|y| y.parse::<f32>().unwrap()))
        .collect::<Vec<_>>();

    let output2 = include_str!("benchmark_data/output_68x120x30.txt")
        .lines()
        .flat_map(|x| x.split_whitespace().map(|y| y.parse::<f32>().unwrap()))
        .collect::<Vec<_>>();
    let details = vec![
        Detection {
            anchors: vec![
                [0.008593750186264515, 0.009722222574055195],
                [0.01614583283662796, 0.016203703358769417],
                [0.03828125074505806, 0.03333333507180214],
            ],
            decode: true,
            decoder: Decoder::ModelPack,
            dtype: DataType::Int8,
            index: 0,
            name: "test".to_string(),
            output_index: 0,
            quantization: None,
            shape: vec![34, 60, 30],
        },
        Detection {
            anchors: vec![
                [0.0018229166744276881, 0.002314814832061529],
                [0.0036458333488553762, 0.004166666883975267],
                [0.0054687499068677425, 0.006018518470227718],
            ],
            decode: true,
            decoder: Decoder::ModelPack,
            dtype: DataType::Int8,
            index: 0,
            name: "test".to_string(),
            output_index: 0,
            quantization: None,
            shape: vec![68, 120, 30],
        },
    ];
    let d = vec![&details[0], &details[1]];
    bencher
        .with_inputs(|| vec![output1.clone(), output2.clone()])
        .bench_local_values(|outputs| {
            let (boxes, scores, nc) = decode_detection_outputs(outputs, &d);
            let mut output_boxes = vec![DetectBox::default(); 50];
            decode_boxes_and_nms(0.5, 0.1, &scores, &boxes, nc, &mut output_boxes, true);
        });
}

#[divan::bench]
fn decode_tensors_benchmark(bencher: divan::Bencher) {
    let output1 = include_str!("benchmark_data/output_34x60x30.txt")
        .lines()
        .flat_map(|x| x.split_whitespace().map(|y| y.parse::<f32>().unwrap()))
        .collect::<Vec<_>>();

    let output2 = include_str!("benchmark_data/output_68x120x30.txt")
        .lines()
        .flat_map(|x| x.split_whitespace().map(|y| y.parse::<f32>().unwrap()))
        .collect::<Vec<_>>();
    let details = vec![
        Detection {
            anchors: vec![
                [0.008593750186264515, 0.009722222574055195],
                [0.01614583283662796, 0.016203703358769417],
                [0.03828125074505806, 0.03333333507180214],
            ],
            decode: true,
            decoder: Decoder::ModelPack,
            dtype: DataType::Int8,
            index: 0,
            name: "test".to_string(),
            output_index: 0,
            quantization: None,
            shape: vec![34, 60, 30],
        },
        Detection {
            anchors: vec![
                [0.0018229166744276881, 0.002314814832061529],
                [0.0036458333488553762, 0.004166666883975267],
                [0.0054687499068677425, 0.006018518470227718],
            ],
            decode: true,
            decoder: Decoder::ModelPack,
            dtype: DataType::Int8,
            index: 0,
            name: "test".to_string(),
            output_index: 0,
            quantization: None,
            shape: vec![68, 120, 30],
        },
    ];
    let d = vec![&details[0], &details[1]];
    bencher
        .with_inputs(|| vec![output1.clone(), output2.clone()])
        .bench_local_values(|outputs| {
            let (_boxes, _scores, _nc) = decode_detection_outputs(outputs, &d);
        });
}

#[divan::bench]
fn decode_boxes_benchmark(bencher: divan::Bencher) {
    let boxes = include_str!("benchmark_data/boxes.txt")
        .lines()
        .flat_map(|x| x.split_whitespace().map(|y| y.parse::<f32>().unwrap()))
        .collect::<Vec<_>>();

    let scores = include_str!("benchmark_data/scores.txt")
        .lines()
        .flat_map(|x| x.split_whitespace().map(|y| y.parse::<f32>().unwrap()))
        .collect::<Vec<_>>();
    bencher.bench_local(|| {
        let _ = decode_boxes(0.05, &scores, &boxes, 5);
    });
}

#[divan::bench]
fn nms_no_class_benchmark(bencher: divan::Bencher) {
    let boxes = include_str!("benchmark_data/boxes.txt")
        .lines()
        .flat_map(|x| x.split_whitespace().map(|y| y.parse::<f32>().unwrap()))
        .collect::<Vec<_>>();

    let scores = include_str!("benchmark_data/scores.txt")
        .lines()
        .flat_map(|x| x.split_whitespace().map(|y| y.parse::<f32>().unwrap()))
        .collect::<Vec<_>>();
    let boxes = decode_boxes(0.0001, &scores, &boxes, 5);
    bencher
        .with_inputs(|| boxes.clone())
        .bench_local_values(|boxes| {
            let _ = nms(0.9, boxes, true);
        });
}

#[divan::bench]
fn nms_with_class_benchmark(bencher: divan::Bencher) {
    let boxes = include_str!("benchmark_data/boxes.txt")
        .lines()
        .flat_map(|x| x.split_whitespace().map(|y| y.parse::<f32>().unwrap()))
        .collect::<Vec<_>>();

    let scores = include_str!("benchmark_data/scores.txt")
        .lines()
        .flat_map(|x| x.split_whitespace().map(|y| y.parse::<f32>().unwrap()))
        .collect::<Vec<_>>();
    let boxes = decode_boxes(0.0001, &scores, &boxes, 5);

    bencher
        .with_inputs(|| boxes.clone())
        .bench_local_values(|boxes| {
            let _ = nms(0.9, boxes, false);
        });
}

#[divan::bench]
fn decodes_nms_benchmark(bencher: divan::Bencher) {
    let boxes = include_str!("benchmark_data/boxes.txt")
        .lines()
        .flat_map(|x| x.split_whitespace().map(|y| y.parse::<f32>().unwrap()))
        .collect::<Vec<_>>();

    let scores = include_str!("benchmark_data/scores.txt")
        .lines()
        .flat_map(|x| x.split_whitespace().map(|y| y.parse::<f32>().unwrap()))
        .collect::<Vec<_>>();
    let mut output_boxes = [DetectBox::default(); 500];
    bencher.bench_local(|| {
        let _ = decode_boxes_and_nms(0.1, 0.1, &scores, &boxes, 5, &mut output_boxes, false);
    });
}

fn main() {
    let env_filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    let stdout_log = tracing_subscriber::fmt::layer()
        .pretty()
        .with_filter(env_filter);

    let env_filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    let journald = match tracing_journald::layer() {
        Ok(journald) => Some(journald.with_filter(env_filter)),
        Err(_) => None,
    };

    let subscriber = Registry::default().with(stdout_log).with(journald);
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
    tracing_log::LogTracer::init().unwrap();

    divan::main();
}
