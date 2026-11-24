use std::{io, path::Path, process::Command};

use serde::Serialize;

use crate::{
    basic_pitch,
    native_pitch::{centered_rms, pcm_samples},
};

const RATE: usize = 44_100;
const FRAME: usize = 2048;
const HOP: usize = 256;

#[derive(Serialize)]
pub(crate) struct Track {
    times: Vec<f64>,
    frequencies: Vec<f64>,
    confidences: Vec<f64>,
    energies: Vec<f64>,
}

fn read_vector(bytes: &mut &[u8]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let (header, data) = bytes
        .split_at_checked(4)
        .ok_or("Missing MELODIA vector header")?;
    let count = u32::from_le_bytes(header.try_into()?) as usize;
    let length = count
        .checked_mul(4)
        .ok_or("MELODIA vector length overflow")?;
    let (samples, remainder) = data
        .split_at_checked(length)
        .ok_or("Incomplete MELODIA frame data")?;
    *bytes = remainder;
    pcm_samples(samples)
}

pub(crate) fn detect(source: &Path) -> Result<Track, Box<dyn std::error::Error>> {
    let executable =
        std::env::var_os("KARAOKE_MELODIA_PATH").unwrap_or_else(|| "karaoke-melodia".into());
    let output = Command::new(executable).arg(source).output().map_err(|error| {
        format!("Unable to start native MELODIA. Install karaoke-melodia or set KARAOKE_MELODIA_PATH: {error}")
    })?;
    if !output.status.success() {
        return Err(format!(
            "Native MELODIA failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    let mut bytes = output.stdout.as_slice();
    let samples = read_vector(&mut bytes)?;
    let pitch = read_vector(&mut bytes)?;
    let confidence = read_vector(&mut bytes)?;
    if !bytes.is_empty() || pitch.len() != confidence.len() {
        return Err("Native MELODIA returned inconsistent frame data".into());
    }

    let count = pitch.len();
    let times = (0..count)
        .map(|index| index as f64 * (HOP as f64 / RATE as f64))
        .collect();
    let confidences: Vec<_> = confidence
        .into_iter()
        .map(|value| {
            if value.is_finite() {
                f64::from(value)
            } else {
                0.0
            }
        })
        .collect();
    let frequencies = pitch
        .into_iter()
        .zip(&confidences)
        .map(|(frequency, &confidence)| {
            if frequency.is_finite() && confidence >= 0.1 {
                f64::from(frequency)
            } else {
                0.0
            }
        })
        .collect();
    let mut energies = centered_rms(&samples, FRAME, HOP);
    energies.resize(count, 0.0);
    Ok(Track {
        times,
        frequencies,
        confidences,
        energies,
    })
}

pub(crate) fn run(source: &Path) -> Result<(), Box<dyn std::error::Error>> {
    serde_json::to_writer(io::stdout().lock(), &detect(source)?)?;
    Ok(())
}

pub(crate) fn fuse(model: &Path, source: &Path) -> Result<(), Box<dyn std::error::Error>> {
    serde_json::to_writer(io::stdout().lock(), &fused_track(model, source)?)?;
    Ok(())
}

pub(crate) fn fused_track(
    model: &Path,
    source: &Path,
) -> Result<basic_pitch::Track, Box<dyn std::error::Error>> {
    let mut base = basic_pitch::detect(model, source, Some(0.0))?;
    let melody = detect(source)?;
    if !melody.times.is_empty() {
        let midi = |hz: f64| (69.0 + 12.0 * (hz.max(1e-6) / 440.0).log2()).round_ties_even();
        for (index, &time) in base.times.iter().enumerate() {
            let right = melody
                .times
                .partition_point(|&other| other < time)
                .min(melody.times.len() - 1);
            let left = right.saturating_sub(1);
            let nearest = if (melody.times[left] - time).abs() <= (melody.times[right] - time).abs()
            {
                left
            } else {
                right
            };
            let frequency = melody.frequencies[nearest];
            if frequency > 0.0
                && (base.frequencies[index] <= 0.0
                    || midi(frequency) != midi(base.frequencies[index]))
            {
                base.frequencies[index] = frequency;
                base.confidences[index] = melody.confidences[nearest];
            }
        }
    }
    Ok(base)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_a_short_melodia_response() {
        assert!(read_vector(&mut [1, 0, 0, 0, 0].as_slice()).is_err());
        assert_eq!(
            read_vector(&mut [1, 0, 0, 0, 0, 0, 128, 63].as_slice()).unwrap(),
            [1.0]
        );
    }
}
