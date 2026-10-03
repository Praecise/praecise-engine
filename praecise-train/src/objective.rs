//! Objectives as per-token weights on `grad(-log p(token))`, and the Muon update.
//!
//! Every token-level objective here has a gradient of the form
//! `sum_t w_t * grad(-log p(y_t))`, where the weight `w_t` depends on the current log-probs but
//! is held constant when differentiating. For one row of logits that gradient is
//! `w_t * (softmax(row) - onehot(y_t))`, which [`dlogits`] builds for the engine's weighted-sum
//! pass. All reductions run in `f64` in a fixed order.

/// `log_softmax(row)[t]` with a fixed summation order.
#[must_use]
pub fn log_softmax_at(row: &[f32], t: usize) -> f64 {
    let mx = row.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(f64::from(x)));
    let sum: f64 = row.iter().map(|&x| (f64::from(x) - mx).exp()).sum();
    f64::from(row[t]) - mx - sum.ln()
}

/// Log-probs of `tokens[i + 1]` under row `i` of `logits` (`tokens.len() - 1` rows of `n_vocab`).
///
/// # Panics
/// When a token is negative or outside the vocabulary.
#[must_use]
pub fn token_logprobs(logits: &[f32], n_vocab: usize, tokens: &[i32]) -> Vec<f64> {
    (0..tokens.len().saturating_sub(1))
        .map(|i| {
            let t = usize::try_from(tokens[i + 1]).expect("token ids are non-negative");
            log_softmax_at(&logits[i * n_vocab..(i + 1) * n_vocab], t)
        })
        .collect()
}

/// `w_i * (softmax(row_i) - onehot(tokens[i + 1]))` for every row, as `f32`.
///
/// # Panics
/// When `weights` does not have one entry per row or a token is outside the vocabulary.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn dlogits(logits: &[f32], n_vocab: usize, tokens: &[i32], weights: &[f64]) -> Vec<f32> {
    let rows = tokens.len() - 1;
    assert_eq!(weights.len(), rows, "one weight per predicted token");
    let mut out = vec![0.0f32; rows * n_vocab];
    for i in 0..rows {
        let w = weights[i];
        if w == 0.0 {
            continue;
        }
        let row = &logits[i * n_vocab..(i + 1) * n_vocab];
        let mx = row.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(f64::from(x)));
        let sum: f64 = row.iter().map(|&x| (f64::from(x) - mx).exp()).sum();
        let t = usize::try_from(tokens[i + 1]).expect("token ids are non-negative");
        for (j, o) in out[i * n_vocab..(i + 1) * n_vocab].iter_mut().enumerate() {
            let p = (f64::from(row[j]) - mx).exp() / sum;
            *o = (w * (p - if j == t { 1.0 } else { 0.0 })) as f32;
        }
    }
    out
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { x.exp() / (1.0 + x.exp()) }
}

/// Direct preference optimization for one pair: `-log sigmoid(beta * (margin - ref_margin))`
/// with `margin = log p(chosen) - log p(rejected)` summed over completion tokens. Returns the
/// loss and the weight on `grad(-log p)` of every chosen completion token; rejected tokens take
/// its negative.
#[must_use]
pub fn dpo(beta: f64, chosen: f64, rejected: f64, ref_chosen: f64, ref_rejected: f64) -> (f64, f64) {
    let z = beta * ((chosen - rejected) - (ref_chosen - ref_rejected));
    // -log sigmoid(z) = softplus(-z), computed without overflow
    let loss = if z > 0.0 { (-z).exp().ln_1p() } else { -z + z.exp().ln_1p() };
    (loss, beta * sigmoid(-z))
}

/// Group-relative advantages: rewards centred on the group mean and scaled by the group standard
/// deviation (population), zero for a group whose rewards are all equal.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn group_advantages(rewards: &[f64]) -> Vec<f64> {
    let n = rewards.len() as f64;
    let mean = rewards.iter().sum::<f64>() / n;
    let var = rewards.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / n;
    let std = var.sqrt();
    rewards.iter().map(|r| if std > 0.0 { (r - mean) / (std + 1e-6) } else { 0.0 }).collect()
}

/// One token of the clipped group-relative policy objective with a KL penalty to a reference
/// policy (the `r - log r - 1` estimator with `r = p_ref / p`). Returns the token's loss and its
/// weight on `grad(-log p)`.
#[must_use]
pub fn grpo_token(logp: f64, sampler_logp: f64, ref_logp: f64, advantage: f64, clip: f64, kl_weight: f64) -> (f64, f64) {
    let ratio = (logp - sampler_logp).exp();
    let clipped = ratio.clamp(1.0 - clip, 1.0 + clip);
    let surrogate = (ratio * advantage).min(clipped * advantage);
    // the gradient flows only where the unclipped term is the minimum
    let active = ratio * advantage <= clipped * advantage;
    let r = (ref_logp - logp).exp();
    let kl = r - (ref_logp - logp) - 1.0;
    let loss = -surrogate + kl_weight * kl;
    let weight = if active { advantage * ratio } else { 0.0 } + kl_weight * (r - 1.0);
    (loss, weight)
}

/// One token of on-policy distillation: the reverse-KL estimate `log p - log q` on a token the
/// student sampled. Returns the estimate and the weight on `grad(-log p)` of its score-function
/// gradient.
#[must_use]
pub fn distill_token(logp: f64, teacher_logp: f64) -> (f64, f64) {
    (logp - teacher_logp, teacher_logp - logp)
}

/// Approximately orthogonalizes a `rows x cols` row-major matrix with quintic Newton-Schulz
/// iterations, as the Muon optimizer does with its momentum: the result has the same singular
/// vectors and singular values close to one.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]
pub fn newton_schulz(m: &[f32], rows: usize, cols: usize, steps: u32) -> Vec<f32> {
    const A: f64 = 3.4445;
    const B: f64 = -4.7750;
    const C: f64 = 2.0315;
    // work on the wide orientation so the Gram matrix is the small one
    let transpose = rows > cols;
    let (r, c) = if transpose { (cols, rows) } else { (rows, cols) };
    let mut x: Vec<f64> = if transpose {
        (0..r * c).map(|k| f64::from(m[(k % c) * cols + k / c])).collect()
    } else {
        m.iter().map(|&v| f64::from(v)).collect()
    };
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt() + 1e-7;
    for v in &mut x {
        *v /= norm;
    }
    for _ in 0..steps {
        // g = x x^T (r x r)
        let mut g = vec![0.0f64; r * r];
        for i in 0..r {
            for j in 0..r {
                g[i * r + j] = (0..c).map(|k| x[i * c + k] * x[j * c + k]).sum();
            }
        }
        // h = B g + C g g
        let mut h = vec![0.0f64; r * r];
        for i in 0..r {
            for j in 0..r {
                let gg: f64 = (0..r).map(|k| g[i * r + k] * g[k * r + j]).sum();
                h[i * r + j] = B * g[i * r + j] + C * gg;
            }
        }
        // x = A x + h x
        let mut next = vec![0.0f64; r * c];
        for i in 0..r {
            for k in 0..c {
                let hx: f64 = (0..r).map(|j| h[i * r + j] * x[j * c + k]).sum();
                next[i * c + k] = A * x[i * c + k] + hx;
            }
        }
        x = next;
    }
    if transpose {
        (0..rows * cols).map(|k| x[(k % cols) * c + k / cols] as f32).collect()
    } else {
        x.into_iter().map(|v| v as f32).collect()
    }
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod tests {
    use super::*;

    #[test]
    fn dlogits_is_the_weighted_cross_entropy_gradient() {
        let n_vocab = 5;
        let logits = [0.3f32, -1.0, 2.0, 0.5, 0.0, 1.0, 1.0, -0.5, 0.2, 0.7];
        let tokens = [0, 2, 4];
        let w = [0.7, -1.3];
        let g = dlogits(&logits, n_vocab, &tokens, &w);
        // finite differences of sum_i w_i * -log p_i
        let f = |l: &[f32]| -> f64 {
            token_logprobs(l, n_vocab, &tokens).iter().zip(&w).map(|(lp, w)| -w * lp).sum()
        };
        for k in 0..logits.len() {
            let mut a = logits;
            let mut b = logits;
            a[k] += 1e-3;
            b[k] -= 1e-3;
            let fd = (f(&a) - f(&b)) / 2e-3;
            assert!((fd - f64::from(g[k])).abs() < 1e-4, "element {k}: fd {fd} vs {}", g[k]);
        }
    }

    #[test]
    fn dpo_weight_is_the_derivative_of_the_loss() {
        let (beta, rc, rr) = (0.3, -4.0, -5.0);
        for (c, r) in [(-3.0, -6.0), (-6.0, -3.0), (-4.0, -5.0)] {
            let (loss, w) = dpo(beta, c, r, rc, rr);
            assert!(loss > 0.0);
            // d loss / d log p(chosen) = -w, d loss / d log p(rejected) = +w
            let fd = (dpo(beta, c + 1e-5, r, rc, rr).0 - dpo(beta, c - 1e-5, r, rc, rr).0) / 2e-5;
            assert!((fd + w).abs() < 1e-6);
            let fd = (dpo(beta, c, r + 1e-5, rc, rr).0 - dpo(beta, c, r - 1e-5, rc, rr).0) / 2e-5;
            assert!((fd - w).abs() < 1e-6);
        }
        // large margins stay finite
        assert!(dpo(1.0, 0.0, -1e4, 0.0, 0.0).0.is_finite() && dpo(1.0, -1e4, 0.0, 0.0, 0.0).0.is_finite());
    }

    #[test]
    fn grpo_token_weight_and_clipping() {
        let (clip, klw) = (0.2, 0.05);
        // on-policy (ratio 1), reference equal: weight is the advantage, no KL
        let (_, w) = grpo_token(-1.0, -1.0, -1.0, 0.8, clip, klw);
        assert!((w - 0.8).abs() < 1e-12);
        // derivative with respect to log p matches -weight where the objective is smooth
        for (lp, adv) in [(-1.05, 0.5), (-0.95, -0.7), (-1.0, 1.2)] {
            let f = |x: f64| grpo_token(x, -1.0, -1.1, adv, clip, klw).0;
            let fd = (f(lp + 1e-6) - f(lp - 1e-6)) / 2e-6;
            let (_, w) = grpo_token(lp, -1.0, -1.1, adv, clip, klw);
            assert!((fd + w).abs() < 1e-5, "lp {lp} adv {adv}: fd {fd} w {w}");
        }
        // a ratio past the clip in the advantage's direction has no surrogate gradient
        let (_, w) = grpo_token(-0.5, -1.0, -0.5, 1.0, clip, 0.0);
        assert!(w.abs() < 1e-12);
        assert_eq!(group_advantages(&[1.0, 1.0, 1.0]), vec![0.0; 3]);
        let a = group_advantages(&[0.0, 1.0]);
        assert!(a[0] < 0.0 && a[1] > 0.0 && (a[0] + a[1]).abs() < 1e-12);
    }

    #[test]
    fn newton_schulz_orthogonalizes() {
        // a signed permutation of diag(3, 1, 0.5, 0.1): the result keeps the singular vectors
        // and maps every singular value into the band the quintic iteration converges to
        let s = [3.0f32, 1.0, 0.5, 0.1];
        for (rows, cols) in [(4usize, 9usize), (9, 4)] {
            let place = |i: usize| ((i * 7 + 1) % rows, (i * 5 + 2) % cols, if i.is_multiple_of(2) { 1.0 } else { -1.0 });
            let mut m = vec![0.0f32; rows * cols];
            for (i, &v) in s.iter().enumerate() {
                let (r, c, sign) = place(i);
                m[r * cols + c] = sign * v;
            }
            let o = newton_schulz(&m, rows, cols, 5);
            for (k, &x) in o.iter().enumerate() {
                match (0..s.len()).map(place).find(|&(r, c, _)| r * cols + c == k) {
                    Some((_, _, sign)) => assert!((0.5..1.3).contains(&(x * sign)), "{rows}x{cols}: singular value {x}"),
                    None => assert!(x.abs() < 1e-6, "{rows}x{cols}: element {k} = {x}"),
                }
            }
        }
    }
}
