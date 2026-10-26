//! Minimum-cost assignment with a cost limit, as ByteTrack matches tracks to
//! detections.
//!
//! A pair costing more than the limit is never matched. Either side may stay
//! unmatched at half the limit each, so the result is the cheapest set of
//! pairs, each under the limit (the extension lapjv's `cost_limit` applies).

const FORBIDDEN: f64 = 1e9;

/// Square Hungarian algorithm with row and column potentials. Returns the
/// column given to each row.
fn hungarian(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    // 1-based arrays, column 0 is the virtual start
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; n + 1];
    let mut row_of = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];
    for i in 1..=n {
        row_of[0] = i;
        let mut j0 = 0;
        let mut min_v = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = row_of[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0;
            for j in 1..=n {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < min_v[j] {
                        min_v[j] = cur;
                        way[j] = j0;
                    }
                    if min_v[j] < delta {
                        delta = min_v[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[row_of[j]] += delta;
                    v[j] -= delta;
                } else {
                    min_v[j] -= delta;
                }
            }
            j0 = j1;
            if row_of[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            row_of[j0] = row_of[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut col_of = vec![0usize; n];
    for j in 1..=n {
        if row_of[j] > 0 {
            col_of[row_of[j] - 1] = j - 1;
        }
    }
    col_of
}

/// Match rows to columns of `cost` (rows x columns, row-major), each pair
/// costing at most `limit`. Returns the matched `(row, column)` pairs, the
/// unmatched rows and the unmatched columns.
#[must_use]
pub fn assign(cost: &[Vec<f64>], cols: usize, limit: f64) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let rows = cost.len();
    if rows == 0 || cols == 0 {
        return (Vec::new(), (0..rows).collect(), (0..cols).collect());
    }
    let n = rows + cols;
    let mut ext = vec![vec![FORBIDDEN; n]; n];
    for (i, row) in cost.iter().enumerate() {
        for (j, &c) in row.iter().enumerate().take(cols) {
            if c <= limit {
                ext[i][j] = c;
            }
        }
        ext[i][cols + i] = limit / 2.0;
    }
    for j in 0..cols {
        ext[rows + j][j] = limit / 2.0;
        for l in 0..rows {
            ext[rows + j][cols + l] = 0.0;
        }
    }
    let col_of = hungarian(&ext);
    let mut matches = Vec::new();
    let mut col_taken = vec![false; cols];
    let mut unmatched_rows = Vec::new();
    for i in 0..rows {
        let j = col_of[i];
        if j < cols && cost[i][j] <= limit {
            matches.push((i, j));
            col_taken[j] = true;
        } else {
            unmatched_rows.push(i);
        }
    }
    let unmatched_cols = (0..cols).filter(|&j| !col_taken[j]).collect();
    (matches, unmatched_rows, unmatched_cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cheapest_assignment_is_found() {
        // greedy would take (0,0) at 0.1 and force (1,1) at 0.9; the optimum is 0.2 + 0.3
        let cost = vec![vec![0.1, 0.2], vec![0.3, 0.9]];
        let (m, r, c) = assign(&cost, 2, 1.0);
        let mut m = m;
        m.sort_unstable();
        assert_eq!(m, vec![(0, 1), (1, 0)]);
        assert!(r.is_empty() && c.is_empty());
    }

    #[test]
    fn pairs_over_the_limit_stay_unmatched() {
        let cost = vec![vec![0.9, 0.95], vec![0.2, 0.85]];
        let (m, r, c) = assign(&cost, 2, 0.8);
        assert_eq!(m, vec![(1, 0)]);
        assert_eq!(r, vec![0]);
        assert_eq!(c, vec![1]);
    }

    #[test]
    fn rectangular_and_empty_inputs() {
        let cost = vec![vec![0.5, 0.1, 0.7]];
        let (m, r, c) = assign(&cost, 3, 0.8);
        assert_eq!(m, vec![(0, 1)]);
        assert!(r.is_empty());
        assert_eq!(c, vec![0, 2]);
        let (m, r, c) = assign(&[], 2, 0.8);
        assert!(m.is_empty() && r.is_empty());
        assert_eq!(c, vec![0, 1]);
    }
}
