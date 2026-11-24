//! Native audio preparation, frame analysis, and pitch inference.

use std::{
    io::{self, Read, Write},
    path::Path,
    process::Command,
};

use serde::Serialize;

const PYIN_SAMPLE_RATE: usize = 22_050;
const PYIN_FRAME: usize = 2048;
const PYIN_HOP: usize = 512;

pub(crate) fn decode_mono(
    source: &Path,
    sample_rate: u32,
    highpass_hz: f64,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    #[derive(serde::Deserialize)]
    struct Stream {
        channels: usize,
        sample_rate: String,
    }

    #[derive(serde::Deserialize)]
    struct Probe {
        streams: Vec<Stream>,
    }

    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels,sample_rate",
            "-of",
            "json",
            "-i",
        ])
        .arg(source)
        .output()?;
    if !probe.status.success() {
        return Err(format!(
            "FFprobe could not read pitch audio: {}",
            String::from_utf8_lossy(&probe.stderr).trim()
        )
        .into());
    }
    let mut probe: Probe = serde_json::from_slice(&probe.stdout)?;
    let stream = probe
        .streams
        .pop()
        .ok_or("Pitch audio has no audio stream")?;
    let source_rate: u32 = stream.sample_rate.parse()?;
    if stream.channels == 0 || sample_rate == 0 || source_rate == 0 {
        return Err("Pitch audio requires channels and a positive sample rate".into());
    }
    // Use equal channel weights, including for multichannel stems.
    let channels = (0..stream.channels)
        .map(|channel| format!("c{channel}"))
        .collect::<Vec<_>>()
        .join("+");
    let downmix = format!("aformat=sample_fmts=flt,pan=mono|c0<{channels}");
    let mut pcm = decode_pcm(
        source,
        &format!("{downmix},aresample={sample_rate}:filter_size=192:cutoff=0.957"),
    )?;
    if pcm.is_empty() && source_rate != sample_rate {
        // The long resampling filter can emit zero samples for very short clips.
        let original = pcm_samples(&decode_pcm(source, &downmix)?)?;
        let count =
            (original.len() as u64 * u64::from(sample_rate)).div_ceil(u64::from(source_rate));
        pcm = (0..count)
            .flat_map(|index| {
                let position = (index as f64 * f64::from(source_rate)) / f64::from(sample_rate);
                let first = position.floor() as usize;
                let last = (first + 1).min(original.len() - 1);
                let fraction = position.fract() as f32;
                (original[first] * (1.0 - fraction) + original[last] * fraction).to_le_bytes()
            })
            .collect();
    }
    if highpass_hz > 0.0 {
        highpass(&mut pcm, sample_rate, highpass_hz);
    }
    Ok(pcm)
}

fn decode_pcm(source: &Path, filter: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let output = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(source)
        .args([
            "-map",
            "0:a:0",
            "-af",
            filter,
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            "pipe:1",
        ])
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "FFmpeg could not decode pitch audio: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    Ok(output.stdout)
}

pub(crate) fn run_decode(
    source: &Path,
    sample_rate: u32,
    highpass_hz: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    io::stdout()
        .lock()
        .write_all(&decode_mono(source, sample_rate, highpass_hz)?)?;
    Ok(())
}

fn highpass(pcm: &mut [u8], sample_rate: u32, cutoff: f64) {
    use std::f64::consts::PI;

    let angle = PI * (2.0 * cutoff / f64::from(sample_rate)).clamp(0.001, 0.49);
    let (sine, cosine) = angle.sin_cos();
    // The two second-order sections of a fourth-order Butterworth filter.
    // Each section uses transposed direct form II and a zero initial state.
    let mut sections = [PI / 8.0, 3.0 * PI / 8.0].map(|pole_angle| {
        let alpha = sine * pole_angle.cos();
        let b0 = (1.0 + cosine) / (2.0 * (1.0 + alpha));
        let a1 = -2.0 * cosine / (1.0 + alpha);
        let a2 = (1.0 - alpha) / (1.0 + alpha);
        (b0, a1, a2, 0.0, 0.0)
    });
    for sample in pcm.chunks_exact_mut(4) {
        let mut value = f64::from(f32::from_le_bytes(sample.try_into().unwrap()));
        for (b0, a1, a2, state1, state2) in &mut sections {
            let filtered = *b0 * value + *state1;
            *state1 = -2.0 * *b0 * value - *a1 * filtered + *state2;
            *state2 = *b0 * value - *a2 * filtered;
            value = filtered;
        }
        sample.copy_from_slice(&(value as f32).to_le_bytes());
    }
}

pub(crate) fn pcm_samples(input: &[u8]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if input.len() % 4 != 0 {
        return Err("Pitch analysis requires f32 PCM input".into());
    }
    Ok(input
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect())
}

pub(crate) fn centered_rms(samples: &[f32], frame: usize, hop: usize) -> Vec<f64> {
    let half = frame / 2;
    // Center each window with zero padding at both ends. Accumulate in f64
    // for stable results even when the track has long silent regions.
    (0..=samples.len() / hop)
        .map(|index| {
            let center = index * hop;
            let start = center.saturating_sub(half);
            let end = (center + half).min(samples.len());
            let power: f64 = samples[start..end]
                .iter()
                .map(|&sample| f64::from(sample).powi(2))
                .sum();
            (power / frame as f64).sqrt()
        })
        .collect()
}

pub(crate) fn run_rms(frame: usize, hop: usize) -> Result<(), Box<dyn std::error::Error>> {
    if frame == 0 || frame % 2 != 0 || hop == 0 {
        return Err("RMS requires a positive even frame length and a positive hop length".into());
    }
    let mut input = Vec::new();
    io::stdin().lock().read_to_end(&mut input)?;
    let samples = pcm_samples(&input)?;
    serde_json::to_writer(io::stdout().lock(), &centered_rms(&samples, frame, hop))?;
    Ok(())
}

fn weak_high_register(frequencies: &[f64], probabilities: &[f64]) -> bool {
    let mut voiced = Vec::new();
    let mut confidence = 0.0;
    for (&frequency, &probability) in frequencies.iter().zip(probabilities) {
        if frequency.is_finite() && frequency > 0.0 {
            voiced.push(frequency);
            confidence += probability;
        }
    }
    if voiced.is_empty() || confidence / voiced.len() as f64 >= 0.35 {
        return false;
    }
    voiced.sort_unstable_by(f64::total_cmp);
    let lower = voiced[(voiced.len() - 1) / 2];
    let upper = voiced[voiced.len() / 2];
    (lower + upper) / 2.0 >= 350.0
}

#[derive(Serialize)]
pub(crate) struct PyinFrames {
    times: Vec<f64>,
    frequencies: Vec<f64>,
    confidences: Vec<f64>,
    energies: Vec<f64>,
    frame: usize,
}

pub(crate) fn run_pyin(
    source: &Path,
    accompaniment: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    serde_json::to_writer(io::stdout().lock(), &detect_pyin(source, accompaniment)?)?;
    Ok(())
}

pub(crate) fn detect_pyin(
    source: &Path,
    accompaniment: Option<&Path>,
) -> Result<PyinFrames, Box<dyn std::error::Error>> {
    let pcm = pcm_samples(&decode_mono(source, PYIN_SAMPLE_RATE as u32, 0.0)?)?;
    let samples: Vec<f64> = pcm.iter().map(|&sample| f64::from(sample)).collect();
    let mut frame = PYIN_FRAME;
    let (mut frequencies, _, mut confidences) = if samples.is_empty() {
        (Vec::new(), Vec::new(), Vec::new())
    } else {
        karaoke_pyin::track(&samples, frame, PYIN_HOP)
    };
    if weak_high_register(&frequencies, &confidences) {
        frame *= 2;
        (frequencies, _, confidences) = karaoke_pyin::track(&samples, frame, PYIN_HOP);
    }
    let times = (0..frequencies.len())
        .map(|index| (index * PYIN_HOP + frame / 2) as f64 / PYIN_SAMPLE_RATE as f64)
        .collect();
    for frequency in &mut frequencies {
        if !frequency.is_finite() {
            *frequency = 0.0;
        }
    }
    let mut energies = centered_rms(&pcm, frame, PYIN_HOP);
    energies.resize(frequencies.len(), 0.0);
    let contour = crate::pitch_contour::prepare(frequencies);
    let accompaniment =
        accompaniment.filter(|_| contour.classes.len() == crate::pitch_contour::CORE_CLASSES);
    let harmonic_class = accompaniment.and_then(|mix| {
        match decode_mono(mix, PYIN_SAMPLE_RATE as u32, 0.0).and_then(|bytes| pcm_samples(&bytes)) {
            Ok(samples) => crate::pitch_contour::harmonic_class(
                &karaoke_pyin::chroma(&samples),
                &contour.classes,
            ),
            Err(error) => {
                eprintln!("Unable to read the accompaniment harmony: {error}");
                None
            }
        }
    });
    let frequencies =
        crate::pitch_contour::finish(contour, accompaniment.is_some(), harmonic_class);
    Ok(PyinFrames {
        times,
        frequencies,
        confidences,
        energies,
        frame,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_uses_centered_windows_and_zero_padding() {
        let rms = centered_rms(&[1.0, 2.0, 3.0, 4.0], 4, 2);
        assert_eq!(rms, [1.25_f64.sqrt(), 7.5_f64.sqrt(), 2.5]);
        assert_eq!(centered_rms(&[2.0], 4, 2), [1.0]);
    }

    #[test]
    fn longer_window_requires_voiced_high_register_and_low_confidence() {
        let voiced = [340.0, 360.0, f64::NAN];
        assert!(weak_high_register(&voiced, &[0.3, 0.2, 0.9]));
        assert!(!weak_high_register(&voiced, &[0.4, 0.4, 0.0]));
        assert!(!weak_high_register(&[330.0], &[0.1]));
        assert!(!weak_high_register(&[f64::NAN], &[0.0]));
    }

    #[test]
    fn highpass_matches_the_fourth_order_butterworth_impulse_response() {
        let mut pcm: Vec<_> = [1.0_f32, 0.0, 0.0, 0.0, 0.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        highpass(&mut pcm, 22_050, 300.0);
        assert_eq!(
            pcm_samples(&pcm).unwrap(),
            [
                0.89428383,
                -0.19973963,
                -0.17713176,
                -0.15605225,
                -0.1364476
            ]
        );
    }

    #[test]
    #[ignore = "requires ffmpeg and ffprobe"]
    fn resampler_keeps_very_short_audio_and_downmixes_channels_equally() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("short clip.wav");
        let mut writer = hound::WavWriter::create(
            &file,
            hound::WavSpec {
                channels: 2,
                sample_rate: 44_100,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..3 {
            writer.write_sample(4096_i16).unwrap();
            writer.write_sample(-4096_i16).unwrap();
        }
        writer.finalize().unwrap();
        let pcm = decode_mono(&file, 22_050, 0.0).unwrap();
        assert_eq!(pcm_samples(&pcm).unwrap(), [0.0, 0.0]);
    }
}
