use serde::Serialize;

#[derive(Serialize)]
pub(crate) struct Boundaries {
    pub(crate) chart: Vec<usize>,
    pub(crate) syllables: Vec<usize>,
}

fn exceptions(word: &str) -> &'static [usize] {
    match word {
        "getting" | "didn't" | "surprise" | "compares" | "mistakes" | "feeling" | "movie"
        | "lover" | "livin'" | "diggin'" | "reaching" | "forty" => &[3],
        "another" | "every" => &[1, 3],
        "settled" | "couldn't" | "breakin'" => &[5],
        "isn't" | "instead" | "regrets" | "making" | "becomes" | "undone" | "losers"
        | "rolling" | "almost" | "fever" | "baby" | "always" | "inside" | "before" | "become"
        | "into" => &[2],
        "reminded" | "seventeen" | "nevermind" | "remember" => &[2, 5],
        "memories" | "finally" => &[2, 4],
        "bittersweet" | "tambourine" => &[3, 6],
        "bringing" | "billie" | "people" | "careful" | "mother" | "happened" | "breaking"
        | "somewhere" | "somethin'" | "champions" | "starting" | "thinking" => &[4],
        "around" => &[1],
        _ => &[],
    }
}

fn lexical_splits(text: &str) -> Vec<usize> {
    let count = text.chars().count();
    if count < 3 {
        return Vec::new();
    }
    let lower = text.to_lowercase();
    let explicit = exceptions(&lower);
    if !explicit.is_empty() {
        return explicit.to_vec();
    }
    if count > 3 && lower.ends_with("n't") {
        return vec![count - 3];
    }
    let mut groups = Vec::new();
    let mut group_start = None;
    let lowered: Vec<_> = lower.chars().collect();
    for (index, &character) in lowered.iter().enumerate() {
        if "aeiouy".contains(character) {
            group_start.get_or_insert(index);
        } else if let Some(start) = group_start.take() {
            groups.push((start, index));
        }
    }
    if let Some(start) = group_start {
        groups.push((start, lowered.len()));
    }
    if groups.len() >= 3 && lower.ends_with("ed") && groups.last().unwrap().1 == count - 1 {
        groups.pop();
    }
    if groups.last().is_some_and(|group| group.1 == count) && lower.ends_with('e') {
        groups.pop();
    }
    if groups.len() < 2 {
        return Vec::new();
    }

    let mut splits: Vec<_> = groups
        .windows(2)
        .filter_map(|pair| (pair[1].0 > pair[0].1).then(|| pair[1].0 - 1))
        .collect();
    let suffix_length = if (count > 5 && (lower.ends_with("thing") || lower.ends_with("times")))
        || (count > 4 && lower.ends_with("ories"))
    {
        Some(5)
    } else if count > 4
        && (lower.ends_with("ing") || lower.ends_with("out") || lower.ends_with("one"))
    {
        Some(3)
    } else {
        None
    };
    if let Some(length) = suffix_length {
        // Chart morphemes can use a different boundary than a vowel nucleus.
        splits.clear();
        if (lower.starts_with("every") && lower.ends_with("thing"))
            || (lower.starts_with("any") && lower.ends_with("one"))
        {
            splits.push(1);
        }
        splits.push(count - length);
    } else {
        if lower.starts_with("un") && count > 4 {
            splits.retain(|&split| split >= 2);
            splits.push(2);
        }
        if count > 5 && lower.ends_with("ted") {
            splits.push(count - 3);
        }
    }
    splits.retain(|&split| split > 0 && split < count);
    splits.sort_unstable();
    splits.dedup();
    splits
}

fn append_marked_splits(boundaries: &mut Vec<usize>, text: &[char], offset: usize) {
    let text: String = text.iter().collect();
    if text.eq_ignore_ascii_case("every") {
        boundaries.push(offset + 3);
    } else {
        boundaries.extend(
            lexical_splits(&text)
                .into_iter()
                .map(|split| offset + split),
        );
    }
}

pub(crate) fn boundaries(text: &str) -> Boundaries {
    let characters: Vec<_> = text.chars().collect();
    let mut syllables = vec![0];
    syllables.extend(lexical_splits(text));
    syllables.push(characters.len());
    if characters.is_empty() {
        return Boundaries {
            chart: vec![0],
            syllables,
        };
    }

    let mut chart = vec![0];
    let mut lexical_start = 0;
    let mut index = 0;
    while index < characters.len() {
        if characters[index] == '~' {
            let start = index;
            index += 1;
            while characters.get(index) == Some(&'~') {
                index += 1;
            }
            append_marked_splits(&mut chart, &characters[lexical_start..start], lexical_start);
            lexical_start = index;
            if start > 0 {
                chart.extend(start..index);
                if characters
                    .get(index)
                    .is_some_and(|c| c.is_alphanumeric() || matches!(c, '\'' | '_'))
                {
                    chart.push(index);
                }
            }
        } else {
            if index > 0 && characters[index] == '-' {
                chart.push(index);
            }
            index += 1;
        }
    }
    if lexical_start > 0 {
        append_marked_splits(&mut chart, &characters[lexical_start..], lexical_start);
    }
    chart.sort_unstable();
    chart.dedup();
    if chart.len() == 1 {
        chart.clone_from(&syllables);
    } else {
        chart.push(characters.len());
    }
    Boundaries { chart, syllables }
}
