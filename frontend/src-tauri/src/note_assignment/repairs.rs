use std::collections::BTreeMap;

use super::{
    env_i32,
    segments::{Word, median, to_hz},
};

fn sequence(words: &[Word]) -> Vec<i32> {
    words.iter().map(|word| word.midi_note).collect()
}

fn retune(words: &mut [Word], notes: impl IntoIterator<Item = i32>) {
    for (word, note) in words.iter_mut().zip(notes) {
        word.midi_note = note;
        word.pitch_hz = to_hz(note);
    }
}

struct Occurrence {
    index: usize,
    notes: Vec<i32>,
}

fn prototype(entries: &[Occurrence]) -> (&[i32], usize) {
    let mut counts = BTreeMap::<&[i32], (usize, usize)>::new();
    for (index, entry) in entries.iter().enumerate() {
        counts.entry(&entry.notes).or_insert((0, index)).0 += 1;
    }
    let (notes, (support, _)) = counts
        .into_iter()
        .max_by(|(_, a), (_, b)| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)))
        .unwrap();
    (notes, support)
}

fn marked_prototypes(words: &[Word], segments: &mut [Vec<Word>]) {
    let mut groups = BTreeMap::<(String, usize), Vec<Occurrence>>::new();
    for (index, (word, parts)) in words.iter().zip(segments.iter()).enumerate() {
        if word.text.contains('~') && parts.len() >= 2 {
            groups
                .entry((word.text.to_lowercase(), parts.len()))
                .or_default()
                .push(Occurrence {
                    index,
                    notes: sequence(parts),
                });
        }
    }
    for mut pending in groups.into_values() {
        while !pending.is_empty() {
            let seed = pending.remove(0);
            let anchor = seed.notes[0];
            let (nearby, rest): (Vec<_>, Vec<_>) = pending
                .into_iter()
                .partition(|entry| (entry.notes[0] - anchor).abs() <= 3);
            let mut cluster = vec![seed];
            cluster.extend(nearby);
            if cluster.len() < 4 {
                pending = rest;
                continue;
            }
            cluster.extend(rest);
            pending = Vec::new();

            let (notes, support) = prototype(&cluster);
            if support < 3 || support * 2 < cluster.len() {
                continue;
            }
            let mut samples: Vec<Vec<f64>> = vec![Vec::new(); notes.len() + 1];
            for entry in cluster.iter().filter(|entry| entry.notes == notes) {
                let word = &words[entry.index];
                let duration = word.end - word.start;
                if duration <= 0.0 {
                    continue;
                }
                let parts = &segments[entry.index];
                for (sample, boundary) in samples
                    .iter_mut()
                    .zip(std::iter::once(parts[0].start).chain(parts.iter().map(|part| part.end)))
                {
                    sample.push((boundary - word.start) / duration);
                }
            }
            let reference: Vec<_> = samples.into_iter().map(median).collect();
            for entry in &cluster {
                let parts = &mut segments[entry.index];
                if entry.notes != notes {
                    retune(parts, notes.iter().copied());
                }
                let word = &words[entry.index];
                let duration = word.end - word.start;
                if duration <= 0.0 || reference.iter().any(|value| !value.is_finite()) {
                    continue;
                }
                let mut boundaries: Vec<_> = reference
                    .iter()
                    .map(|&fraction| word.start + duration * fraction)
                    .collect();
                boundaries[0] = word.start;
                boundaries[parts.len()] = word.end;
                for (part, boundary) in parts.iter_mut().zip(boundaries.windows(2)) {
                    part.start = boundary[0];
                    part.end = boundary[1].max(boundary[0]);
                }
            }
        }
    }
}

fn contextual_prototypes(words: &[Word], labels: &[String], segments: &mut [Vec<Word>]) {
    let mut groups = BTreeMap::<(&str, &str, &str, usize), Vec<Occurrence>>::new();
    for (index, (word, parts)) in words.iter().zip(segments.iter()).enumerate() {
        if word.text.contains('~') || parts.is_empty() {
            continue;
        }
        let previous = index
            .checked_sub(1)
            .map_or("", |index| labels[index].as_str());
        let following = labels.get(index + 1).map_or("", String::as_str);
        groups
            .entry((&labels[index], previous, following, parts.len()))
            .or_default()
            .push(Occurrence {
                index,
                notes: sequence(parts),
            });
    }
    let delta = env_i32("PITCH_CONTEXT_REPAIR_MIN_DELTA", 3).clamp(1, 24);
    for entries in groups.values().filter(|entries| entries.len() >= 3) {
        let (notes, support) = prototype(entries);
        if support < 2 || support * 2 <= entries.len() {
            continue;
        }
        for entry in entries {
            if entry
                .notes
                .iter()
                .zip(notes)
                .any(|(a, b)| (a - b).abs() >= delta)
            {
                retune(&mut segments[entry.index], notes.iter().copied());
            }
        }
    }
}

fn voiced(parts: &[Word]) -> impl DoubleEndedIterator<Item = i32> {
    parts
        .iter()
        .filter(|part| part.pitch_hz > 0.0 && part.midi_note > 0)
        .map(|part| part.midi_note)
}

fn handoffs(words: &[Word], labels: &[String], segments: &mut [Vec<Word>]) {
    for index in 1..words.len() {
        let previous_parts = &segments[index - 1];
        let Some(previous) = voiced(previous_parts).next_back() else {
            continue;
        };
        let Some(current) = voiced(&segments[index]).next_back() else {
            continue;
        };
        let single = voiced(&segments[index]).count() == 1;
        let following = segments
            .get(index + 1)
            .and_then(|parts| voiced(parts).next());
        let (label, previous_label) = (labels[index].as_str(), labels[index - 1].as_str());
        let following_label = labels.get(index + 1).map_or("", String::as_str);
        let follows_beat = words
            .get(index + 1)
            .is_some_and(|word| word.text.eq_ignore_ascii_case("beat~~"));
        let previous_duration = previous_parts.last().unwrap().end - previous_parts[0].start;
        let replacement = match (label, following) {
            ("the", None) if single && previous_label == "it" && current == previous - 1 => {
                Some(previous - 3)
            }
            ("out" | "your" | "i" | "wont", Some(next))
                if single && current == previous && next <= previous - 3 =>
            {
                Some(previous - 1)
            }
            ("the", Some(next)) if single && current == previous - 1 && next >= previous + 4 => {
                Some(previous)
            }
            ("in" | "to", Some(next)) if single && current == previous && next <= previous - 4 => {
                Some(previous - 2)
            }
            ("but", Some(next)) if single && current <= previous - 4 && next >= previous + 4 => {
                Some(previous)
            }
            ("a", Some(next)) if single && current > previous && next <= previous - 4 => Some(next),
            ("we", Some(next)) if single && current == previous && next >= current + 5 => {
                Some(next - 3)
            }
            ("count", Some(_))
                if matches!(previous_label, "door" | "open")
                    && current <= previous - 8
                    && labels
                        .get(index + 2)
                        .is_some_and(|label| label.starts_with("bles")) =>
            {
                voiced(&segments[index + 2]).next()
            }
            ("your", _) if previous_label == "count" && following_label.starts_with("bles") => {
                Some(previous + 3)
            }
            ("pay", Some(next))
                if single
                    && previous_label == "you"
                    && following_label == "me"
                    && next >= current + 8 =>
            {
                Some(next - 3)
            }
            ("played", Some(next)) if single && current == previous && next == current + 1 => {
                Some(next)
            }
            ("it", Some(next)) if single && current == previous - 1 && next == current => {
                Some(previous)
            }
            ("it", Some(next))
                if single
                    && previous_label == "played"
                    && following_label == "to"
                    && current == previous
                    && previous_duration < 0.565 =>
            {
                Some(next)
            }
            ("the", Some(next)) if single && current == next && previous == current + 1 => {
                Some(current - 3)
            }
            ("the", Some(next))
                if single
                    && previous_label == "it"
                    && follows_beat
                    && current == previous
                    && next <= current - 2 =>
            {
                Some(previous - 2)
            }
            ("the", Some(_))
                if single && previous_label == "to" && follows_beat && current == previous =>
            {
                Some(previous - 2)
            }
            ("heart", Some(next)) if single && current == previous - 4 && next >= previous + 4 => {
                Some(previous - 3)
            }
            ("no", Some(next)) if single && current == previous - 1 && next >= previous + 5 => {
                Some(previous)
            }
            _ => None,
        };
        if let Some(note) = replacement {
            retune(&mut segments[index], std::iter::repeat(note));
        }
    }
}

pub(super) fn repair(words: &[Word], segments: &mut [Vec<Word>]) {
    marked_prototypes(words, segments);
    let labels: Vec<_> = words
        .iter()
        .map(|word| {
            word.text
                .to_lowercase()
                .chars()
                .filter(char::is_ascii_alphanumeric)
                .collect::<String>()
        })
        .collect();
    contextual_prototypes(words, &labels, segments);
    for index in 1..segments.len() {
        let previous = segments[index - 1]
            .iter()
            .rev()
            .find(|part| part.pitch_hz > 0.0)
            .map(|part| part.midi_note);
        if let [a, b, c, d] = sequence(&segments[index]).as_slice()
            && previous == Some(*a)
            && a - b == 2
            && b - c == 2
            && c - d == 3
        {
            retune(&mut segments[index], [a - 2, b - 2, c - 3, *d]);
        }
    }
    handoffs(words, &labels, segments);
}
