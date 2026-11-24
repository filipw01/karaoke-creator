//! An opt-in ONNX forward pass for Basic Pitch. Python retains preprocessing
//! and karaoke-specific postprocessing for direct numerical comparisons.

use std::{
    io::{self, BufReader, BufWriter, Read, Write},
    path::Path,
};

use ort::{session::Session, value::Tensor};
use serde::Serialize;

const SAMPLES: usize = 43_844;

#[derive(Serialize)]
struct Output<'a> {
    name: &'a str,
    shape: Vec<i64>,
    values: Vec<f32>,
}

pub(crate) fn run(model: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let threads = |key| {
        std::env::var(key)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(1)
    };
    if let Some(runtime) = std::env::var_os("ORT_DYLIB_PATH") {
        ort::init_from(runtime)?.commit();
    }
    let mut session = Session::builder()?
        .with_intra_threads(threads("BASIC_PITCH_INTRA_THREADS"))?
        .with_inter_threads(threads("BASIC_PITCH_INTER_THREADS"))?
        .commit_from_file(model)?;

    let mut input = BufReader::new(io::stdin().lock());
    let mut output = BufWriter::new(io::stdout().lock());
    let mut encoded = vec![0u8; SAMPLES * 4];
    while input.read(&mut encoded[..1])? != 0 {
        input.read_exact(&mut encoded[1..])?;
        let window: Vec<f32> = encoded
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let tensor = Tensor::from_array(([1usize, SAMPLES, 1], window))?;
        let results = session.run(ort::inputs![tensor])?;
        let tensors = results
            .iter()
            .map(|(name, value)| {
                let (shape, values) = value.try_extract_tensor::<f32>()?;
                Ok(Output {
                    name,
                    shape: shape.to_vec(),
                    values: values.to_vec(),
                })
            })
            .collect::<ort::Result<Vec<_>>>()?;
        serde_json::to_writer(&mut output, &tensors)?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    Ok(())
}

pub(crate) fn run_pyin(frame: usize, hop: usize) -> Result<(), Box<dyn std::error::Error>> {
    if !matches!(frame, 2048 | 4096) || hop == 0 || hop > frame {
        return Err("Invalid pYIN frame or hop length".into());
    }
    let mut input = Vec::new();
    io::stdin().lock().read_to_end(&mut input)?;
    if input.is_empty() || input.len() % 4 != 0 {
        return Err("pYIN requires nonempty f32 PCM input".into());
    }
    let samples: Vec<f64> = input
        .chunks_exact(4)
        .map(|bytes| f64::from(f32::from_le_bytes(bytes.try_into().unwrap())))
        .collect();
    let pitch = karaoke_pyin::track(&samples, frame, hop);
    serde_json::to_writer(io::stdout().lock(), &pitch)?;
    Ok(())
}
