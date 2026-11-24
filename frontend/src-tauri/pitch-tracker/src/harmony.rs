use std::f32::consts::{PI, TAU};

use rustfft::{FftPlanner, num_complex::Complex};

const SAMPLE_RATE: f32 = 22_050.0;
const HOP: usize = 512;
const BINS: usize = 36;
const OCTAVES: usize = 7;
const C1: f32 = 32.703_197;

fn tuning(samples: &[f32]) -> f32 {
    const WINDOW: usize = 2048;
    let fft = FftPlanner::new().plan_fft_forward(WINDOW);
    let mut scratch = vec![Complex::default(); fft.get_inplace_scratch_len()];
    let mut spectrum = vec![Complex::default(); WINDOW];
    let window: Vec<_> = (0..WINDOW)
        .map(|i| 0.5 - 0.5 * (TAU * i as f32 / WINDOW as f32).cos())
        .collect();
    let mut magnitudes = vec![0.0; WINDOW / 2];
    let mut peaks = Vec::new();
    for center in (0..samples.len()).step_by(HOP) {
        for (index, value) in spectrum.iter_mut().enumerate() {
            let sample = (center + index)
                .checked_sub(WINDOW / 2)
                .and_then(|offset| samples.get(offset))
                .copied()
                .unwrap_or(0.0);
            *value = Complex::new(sample * window[index], 0.0);
        }
        fft.process_with_scratch(&mut spectrum, &mut scratch);
        for (magnitude, value) in magnitudes.iter_mut().zip(&spectrum) {
            *magnitude = value.norm();
        }
        let minimum = 0.1 * magnitudes.iter().copied().fold(0.0, f32::max);
        let step = SAMPLE_RATE / WINDOW as f32;
        for (index, peak) in magnitudes.windows(3).enumerate() {
            let [left, middle, right] = [peak[0], peak[1], peak[2]];
            let bin = (index + 1) as f32;
            if !(150.0..4000.0).contains(&(bin * step))
                || middle <= minimum
                || middle <= left
                || middle < right
            {
                continue;
            }
            let shift = 0.5 * (left - right) / (left - 2.0 * middle + right);
            let frequency = (bin + shift) * step;
            let deviation = (BINS as f32 * (frequency / 440.0).log2() + 0.5).rem_euclid(1.0);
            let bucket = ((deviation * 100.0) as usize).min(99);
            let magnitude = middle - 0.25 * (left - right) * shift;
            peaks.push((magnitude, bucket));
        }
    }
    if peaks.is_empty() {
        return 0.0;
    }
    let middle = peaks.len() / 2;
    peaks.select_nth_unstable_by(middle, |a, b| a.0.total_cmp(&b.0));
    let lower = peaks[middle].0;
    let mut histogram = [0usize; 100];
    for (strength, bucket) in peaks {
        if strength >= lower {
            histogram[bucket] += 1;
        }
    }
    let bucket = (0..100)
        .min_by_key(|&bucket| (std::cmp::Reverse(histogram[bucket]), bucket))
        .unwrap();
    bucket as f32 / 100.0 - 0.5
}

fn decimate(samples: &[f32]) -> Vec<f32> {
    // A low-pass Blackman-windowed sinc prevents high octaves from aliasing
    // into the next octave's frequency bands.
    const HALF: usize = 31;
    const CUTOFF: f32 = 0.24;
    let kernel: Vec<_> = (0..=2 * HALF)
        .map(|index| {
            let phase = PI * (index as f32 - HALF as f32);
            let window =
                0.42 + 0.5 * (phase / HALF as f32).cos() + 0.08 * (2.0 * phase / HALF as f32).cos();
            let impulse = if index == HALF {
                2.0 * CUTOFF
            } else {
                (2.0 * CUTOFF * phase).sin() / phase
            };
            window * impulse
        })
        .collect();
    let scale = 2.0_f32.sqrt() / kernel.iter().sum::<f32>();
    (0..samples.len())
        .step_by(2)
        .map(|center| {
            let start = center.saturating_sub(HALF);
            let end = (center + HALF + 1).min(samples.len());
            let offset = HALF.saturating_sub(center);
            samples[start..end]
                .iter()
                .zip(&kernel[offset..])
                .map(|(sample, weight)| sample * weight)
                .sum::<f32>()
                * scale
        })
        .collect()
}

/// Mean of the peak-normalized constant-Q chroma columns, starting at C.
pub fn chroma(samples: &[f32]) -> [f32; 12] {
    if samples.len() < SAMPLE_RATE as usize {
        return [0.0; 12];
    }
    let mut frames = vec![[0.0; 12]; 1 + samples.len() / HOP];
    let correction = tuning(samples);
    let bandwidth =
        (2.0_f32.powf(2.0 / BINS as f32) - 1.0) / (2.0_f32.powf(2.0 / BINS as f32) + 1.0);
    let mut octave_samples = decimate(samples);
    let mut sample_rate = SAMPLE_RATE / 2.0;
    let mut hop = HOP / 2;
    for octave in (0..OCTAVES).rev() {
        for bin in 0..BINS {
            let frequency =
                C1 * 2.0_f32.powf(octave as f32 + (bin as f32 + correction) / BINS as f32);
            let length = sample_rate / (bandwidth * frequency);
            let half = (length / 2.0).floor() as usize;
            let weights: Vec<_> = (0..=2 * half)
                .map(|index| {
                    let angle = TAU * index as f32;
                    let hann = 0.5 - 0.5 * (angle / (2 * half + 1) as f32).cos();
                    let (sine, cosine) = (angle * frequency / sample_rate).sin_cos();
                    (hann * cosine, hann * sine)
                })
                .collect();
            let norm = length.sqrt() * 2.0 / weights.len() as f32;
            let class = ((bin + 1) / 3) % 12;
            for (frame, chroma) in frames.iter_mut().enumerate() {
                let center = frame * hop;
                let start = center.saturating_sub(half).min(octave_samples.len());
                let end = (center + half + 1).min(octave_samples.len());
                let offset = half.saturating_sub(center);
                let mut real = 0.0;
                let mut imag = 0.0;
                for (&sample, &(cosine, sine)) in
                    octave_samples[start..end].iter().zip(&weights[offset..])
                {
                    real += sample * cosine;
                    imag += sample * sine;
                }
                chroma[class] += real.hypot(imag) * norm;
            }
        }
        if octave > 0 {
            octave_samples = decimate(&octave_samples);
            sample_rate /= 2.0;
            hop /= 2;
        }
    }
    let mut result = [0.0; 12];
    for frame in &frames {
        let peak = frame.iter().copied().fold(0.0, f32::max);
        if peak > 0.0 {
            for (total, score) in result.iter_mut().zip(frame) {
                *total += score / peak;
            }
        }
    }
    result.map(|value| value / frames.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuned_octaves_land_in_the_same_pitch_class() {
        let samples: Vec<_> = (0..2 * SAMPLE_RATE as usize)
            .map(|index| {
                let position = TAU * index as f32 / SAMPLE_RATE;
                (position * 221.0).sin() + (position * 442.0).sin()
            })
            .collect();
        let scores = chroma(&samples);
        assert!(scores[9] > 0.98);
        assert!(
            scores
                .iter()
                .enumerate()
                .all(|(class, score)| class == 9 || *score < 0.3)
        );
        assert_eq!(chroma(&vec![0.0; SAMPLE_RATE as usize]), [0.0; 12]);
    }
}
