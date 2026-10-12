//! Drafting from a sequence's own tokens.
//!
//! The last few tokens a sequence holds are looked up earlier in the same
//! sequence, prompt and output alike, and the tokens that followed the most
//! recent earlier occurrence are proposed as the draft. Replies that quote
//! their prompt, edit code, fill a fixed structure or repeat a name draft long
//! accepted blocks this way, with no second model, no draft context and no
//! memory beyond the tokens the sequence already keeps.
//!
//! A draft never decides a token. The target scores every drafted position in
//! the step that verifies it and its sampler picks each token exactly as plain
//! decoding would, so a poor draft costs only the rows it occupied in that step.
//!
//! Lookup is a pure function of the sequence's tokens: the same sequence always
//! drafts the same block, whatever else the engine is serving.

/// Longest suffix looked up.
pub const NGRAM_MAX: usize = 4;
/// Shortest suffix looked up. A single repeated token is too weak a signal to
/// pay for the rows its draft would take.
pub const NGRAM_MIN: usize = 2;

/// Propose up to `max_draft` tokens to follow `history`.
///
/// Tries the longest suffix first, from [`NGRAM_MAX`] tokens down to
/// [`NGRAM_MIN`], and for each takes the most recent earlier position where it
/// occurs. Returns the tokens that followed there, or nothing when no suffix
/// recurs.
pub fn draft<T: Copy + PartialEq>(history: &[T], max_draft: usize) -> Vec<T> {
    draft_with(history, max_draft, NGRAM_MIN, NGRAM_MAX)
}

/// [`draft`] with an explicit suffix range.
pub fn draft_with<T: Copy + PartialEq>(history: &[T], max_draft: usize, ngram_min: usize, ngram_max: usize) -> Vec<T> {
    let len = history.len();
    if max_draft == 0 || ngram_min == 0 || len < ngram_min + 1 {
        return Vec::new();
    }
    for n in (ngram_min..=ngram_max.min(len - 1)).rev() {
        let key = &history[len - n..];
        // Every start whose match ends before the suffix itself, newest first.
        let mut start = len - n;
        while start > 0 {
            start -= 1;
            if history[start..start + n] == *key {
                let from = start + n;
                let to = (from + max_draft).min(len);
                if from < to {
                    return history[from..to].to_vec();
                }
            }
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_without_a_repeat() {
        assert!(draft(&[1, 2, 3, 4, 5], 4).is_empty());
        assert!(draft::<i32>(&[], 4).is_empty());
        assert!(draft(&[7], 4).is_empty());
    }

    #[test]
    fn continues_the_most_recent_occurrence() {
        // "1 2" occurs twice before the suffix; the later one is followed by 9.
        let h = [1, 2, 3, 1, 2, 9, 8, 1, 2];
        assert_eq!(draft(&h, 2), vec![9, 8]);
    }

    #[test]
    fn prefers_the_longest_suffix() {
        // "2 3" alone last continues with 7, but "1 2 3" continues with 5.
        let h = [1, 2, 3, 5, 6, 0, 2, 3, 7, 4, 1, 2, 3];
        assert_eq!(draft(&h, 2), vec![5, 6]);
    }

    #[test]
    fn caps_the_block() {
        let h = [1, 2, 3, 4, 5, 6, 7, 1, 2];
        assert_eq!(draft(&h, 3), vec![3, 4, 5]);
        assert!(draft(&h, 0).is_empty());
    }

    #[test]
    fn a_draft_may_run_into_the_suffix() {
        // A loop of period two drafts its own continuation.
        let h = [4, 5, 4, 5, 4, 5];
        assert_eq!(draft(&h, 3), vec![4, 5]);
    }

    #[test]
    fn single_token_repeats_do_not_draft() {
        assert!(draft(&[3, 8, 1, 3], 4).is_empty());
    }

    #[test]
    fn deterministic() {
        let h: Vec<i32> = (0..500).map(|i| (i * 7919 % 13) as i32).collect();
        assert_eq!(draft(&h, 6), draft(&h, 6));
    }
}
