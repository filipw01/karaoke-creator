use std::collections::BTreeMap;

type Run = (usize, usize, i32);

pub(crate) struct ForcedRuns {
    pub(super) notes: Vec<i32>,
    pub(super) voiced: Vec<bool>,
    pub(super) weights: Vec<Option<f64>>,
    pub(super) times: Vec<f64>,
    pub(super) target: usize,
    pub(super) minimum: usize,
    pub(super) frame_step: f64,
    pub(super) duration: f64,
    pub(super) marked: bool,
}

fn partition(
    notes: &[i32],
    voiced: &[bool],
    weights: &[Option<f64>],
    target: usize,
    minimum: usize,
) -> Vec<Run> {
    let count = notes.len();
    let minimum = minimum.max(1);
    let target = target.min(count / minimum);
    if target <= 1 {
        return Vec::new();
    }
    let maximum = if target <= 4 {
        (2 * count).div_ceil(target).max(minimum)
    } else {
        count
    };
    let mut costs = vec![(f64::INFINITY, 0); count * maximum];
    for start in 0..count {
        let mut totals: BTreeMap<i32, (f64, usize)> = BTreeMap::new();
        for end in start..(start + maximum).min(count) {
            if voiced[end] && notes[end] > 0 {
                let score = weights[end].unwrap_or(0.0).max(1e-6);
                totals.entry(notes[end]).or_insert((0.0, end)).0 += score;
            }
            if let Some((&mode, _)) = totals
                .iter()
                .max_by(|(_, a), (_, b)| a.0.total_cmp(&b.0).then_with(|| b.1.cmp(&a.1)))
            {
                let cost = (start..=end)
                    .filter(|&i| voiced[i] && notes[i] > 0)
                    .map(|i| {
                        weights[i].unwrap_or(0.0).max(1e-6) * f64::from((notes[i] - mode).abs())
                    })
                    .sum::<f64>();
                costs[start * maximum + end - start] = (cost, mode);
            }
        }
    }

    let mut paths = vec![(f64::INFINITY, 0); (target + 1) * count];
    for end in minimum - 1..maximum {
        paths[count + end].0 = costs[end].0;
    }
    for parts in 2..=target {
        for end in parts * minimum - 1..count {
            let first = ((parts - 1) * minimum - 1).max(end.saturating_sub(maximum));
            let last = end - minimum;
            for split in first..=last {
                let candidate = paths[(parts - 1) * count + split].0
                    + costs[(split + 1) * maximum + end - split - 1].0;
                let path = &mut paths[parts * count + end];
                if candidate < path.0 {
                    *path = (candidate, split);
                }
            }
        }
    }

    let mut end = count - 1;
    if !paths[target * count + end].0.is_finite() {
        return Vec::new();
    }
    let mut runs = Vec::with_capacity(target);
    for parts in (1..=target).rev() {
        let split = paths[parts * count + end].1;
        let start = if parts > 1 { split + 1 } else { 0 };
        runs.push((start, end, costs[start * maximum + end - start].1));
        end = split;
    }
    runs.reverse();
    runs
}

fn stable_modes(runs: &mut [Run], marked: bool) {
    let mut values: Vec<_> = runs.iter().map(|run| run.2).collect();
    let mut counts = BTreeMap::new();
    for &value in &values {
        *counts.entry(value).or_insert(0) += 1;
    }
    for index in 0..values.len() {
        let value = values[index];
        if counts[&value] != 1 {
            continue;
        }
        let neighbors = index
            .checked_sub(1)
            .and_then(|i| values.get(i))
            .into_iter()
            .chain(values.get(index + 1));
        let replacement = [value - 12, value + 12]
            .into_iter()
            .filter(|candidate| counts.get(candidate).copied().unwrap_or(0) >= 2)
            .min_by_key(|candidate| {
                neighbors
                    .clone()
                    .map(|neighbor| (candidate - neighbor).abs())
                    .min()
                    .unwrap_or(12)
            });
        if let Some(replacement) = replacement {
            values[index] = replacement;
        }
    }

    match values.as_mut_slice() {
        [first, second] => match *second - *first {
            1 => *second = *first + 3,
            2 => *first = *second,
            -4 => *first = *second + 2,
            interval if interval <= -12 => {
                *first = *second + 5;
                *second += 2;
            }
            _ => {}
        },
        [first, second, third] => {
            let first_step = *second - *first;
            let second_step = *third - *second;
            if marked
                && first_step >= 0
                && (0..=2).contains(&second_step)
                && (2..=6).contains(&(*third - *first))
            {
                *first = *third - 5;
                *second = *third - 2;
            } else if (-3..=-1).contains(&first_step)
                && (-5..=-3).contains(&second_step)
                && (5..=7).contains(&(*first - *third))
            {
                *third = *second - 3;
            } else if first_step == 0 && second_step == -3 {
                *second = *first - 1;
            } else if first_step == 4 && second_step == 0 {
                *first = *second;
            } else if first_step == 12 && second_step == -3 {
                *first = *second;
                *second -= 1;
            } else if first_step == -4 && second_step == -3 {
                *first = *second + 2;
            }
        }
        [first, second, third, fourth] => {
            if first == fourth && (*second - *third).abs() <= 2 && (*second).max(*third) > *first {
                let plateau = (*second).max(*third);
                *second = plateau;
                *third = plateau;
            }
            if first == fourth && *second == *first + 2 && (3..=5).contains(&(*third - *first)) {
                *third = *first + 4;
            }
            let drop = *first - *fourth;
            if *first > *second
                && *second >= *third
                && *third >= *fourth
                && (6..=8).contains(&drop)
                && (*second - *third >= 3 || third == fourth)
            {
                *second = *first - if drop <= 7 { 2 } else { 3 };
                *third = *second - 2;
            }
        }
        [first, second, third, fourth, fifth]
            if *first - *third >= 10
                && *second - *third >= 10
                && *fourth >= *third - 1
                && *third > *fifth =>
        {
            *first = *third + 5;
            *second = *third + 2;
            *fourth = *third - 2;
            *fifth = *third - 5;
        }
        _ => {}
    }
    let last = values.len() - 1;
    if values.len() >= 3 && (values[last] - values[0]).abs() == 12 {
        values[last] = values[0];
    }
    for (run, note) in runs.iter_mut().zip(values) {
        run.2 = note;
    }
}

impl ForcedRuns {
    pub(crate) fn resolve(self) -> Result<Vec<Run>, String> {
        let count = self.notes.len();
        if self.voiced.len() != count || self.weights.len() != count || self.times.len() != count {
            return Err("Forced pitch partition requires matching frame arrays".into());
        }
        let mut runs = partition(
            &self.notes,
            &self.voiced,
            &self.weights,
            self.target,
            self.minimum,
        );
        if runs.is_empty() {
            return Ok(runs);
        }
        stable_modes(&mut runs, self.marked);
        let tail = self
            .minimum
            .max((0.16 / self.frame_step).round_ties_even() as usize);
        if self.frame_step > 0.0 {
            for index in 0..runs.len() - 1 {
                let (left, right) = (runs[index], runs[index + 1]);
                if left.2 == right.2 && right.0 == left.1 + 1 && right.1 - left.0 + 1 >= 2 * tail {
                    let cut = right.1 - tail + 1;
                    runs[index].1 = cut - 1;
                    runs[index + 1].0 = cut;
                }
            }
        }
        let duration = |run: &Run| self.times[run.1] - self.times[run.0];
        match runs.as_mut_slice() {
            [left, right] => {
                if duration(left) <= 0.15
                    && (duration(right) >= 0.10 || self.duration >= 0.30)
                    && right.2 > 0
                    && (left.2 == right.2 || left.2 == right.2 - 3)
                {
                    left.2 = right.2 - 1;
                }
                if self.duration >= 0.90
                    && left.2 - right.2 == 2
                    && duration(left) >= 0.35
                    && duration(right) >= 0.35
                {
                    left.2 += 2;
                }
            }
            [first, second, third]
                if !self.marked
                    && first.2 == second.2
                    && first.2 == third.2
                    && duration(first) <= 0.11
                    && duration(second) <= 0.21 =>
            {
                first.2 -= 2;
            }
            _ => {}
        }
        Ok(runs)
    }
}
