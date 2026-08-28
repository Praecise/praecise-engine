//! N-gram self-speculation: drafting from token history, with no draft model.
//!
//! Speculative decoding normally needs a second set of weights to propose
//! tokens. This does not. It keeps a map from an n-gram of recent tokens to the
//! token that followed it last time, and proposes that continuation. Text
//! repeats itself — a variable name recurs, a phrase comes back, an agent loop
//! re-reads the same file — and every repetition is a free draft.
//!
//! It costs no GPU memory and no second model load, which is what makes it
//! belong in the acceleration layer rather than in a backend: the only input is
//! the token sequence, which every backend already has.
//!
//! ## Why a bounded hash map rather than a suffix tree
//!
//! A suffix tree gives exact longest-match at the cost of unbounded growth and
//! pointer chasing. This uses a fixed-capacity map keyed on the last `n`
//! tokens, which bounds memory and makes lookup a single probe. llama.cpp's
//! equivalent (`ngram-mod`) documents 0.703 acceptance against 0.576 for a
//! simpler variant; a shared, reinforced pool is why.
//!
//! ## Cross-request reuse
//!
//! [`NgramCache`] is deliberately not tied to a request. Agentic loops, code
//! editing and RL rollouts repeat across turns, and a cache that survives the
//! request is the reason to put this in a layer above the backend — no single
//! runtime is positioned to keep state between calls.
//!
//! Sharing one across *users* is a different matter: token history is user
//! content, and a shared cache leaks it through drafted tokens and through
//! timing. Scope a cache to one tenant.

use std::collections::HashMap;

/// Shortest key used: how many recent tokens must match before we trust the
/// continuation that followed them.
///
/// Two is the useful floor. One is a unigram — "what usually follows this
/// token" — which is mostly noise. Longer keys are more selective but fire less
/// often, so the cache keeps several orders and prefers the longest.
pub const MIN_NGRAM: usize = 2;

/// Longest key tried. Beyond this the extra selectivity stops paying for the
/// extra lookups: a 4-token match is already specific enough that the token
/// after it is near-deterministic.
pub const MAX_NGRAM: usize = 4;

/// Default cap on distinct keys held. Each entry is small, so this is on the
/// order of a megabyte — chosen to stay far below the cost of a draft model,
/// which is the whole point of this path.
pub const DEFAULT_CAPACITY: usize = 1 << 16;

/// One remembered continuation, plus how often it has held.
///
/// The count is what lets a stable continuation outrank a one-off. Without it
/// the most recent write always wins, and a single unusual completion evicts a
/// pattern that had been holding for thousands of tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Continuation {
    token: i32,
    hits: u32,
}

/// A bounded map from recent-token n-grams to the token that followed them.
///
/// Not tied to a request: see the module docs on cross-request reuse, and on
/// why a cache must not be shared between tenants.
#[derive(Debug, Clone)]
pub struct NgramCache {
    /// Keyed by the n-gram itself rather than by a hash of it, so a collision
    /// cannot silently propose a token from an unrelated context. Drafts are
    /// verified by the target, so a collision could not corrupt output — but it
    /// would waste a verify slot, and those are the budget being spent here.
    table: HashMap<Vec<i32>, Continuation>,
    capacity: usize,
}

impl Default for NgramCache {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }
}

impl NgramCache {
    /// An empty cache holding at most `capacity` distinct n-grams.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            table: HashMap::new(),
            capacity: capacity.max(1),
        }
    }

    /// How many n-grams are currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Whether the cache has learned anything yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Learn from a token sequence: for every order and position, record which
    /// token followed.
    ///
    /// Call this with the prompt before generating, and with accepted tokens
    /// afterwards. Feeding it *rejected* drafts would teach the cache the
    /// target's mistakes, so callers must only observe what was accepted.
    pub fn observe(&mut self, tokens: &[i32]) {
        for n in MIN_NGRAM..=MAX_NGRAM {
            if tokens.len() <= n {
                continue;
            }
            for w in tokens.windows(n + 1) {
                let (key, next) = w.split_at(n);
                self.record(key.to_vec(), next[0]);
            }
        }
    }

    /// Record one key -> token observation, reinforcing or decaying.
    fn record(&mut self, key: Vec<i32>, token: i32) {
        match self.table.get_mut(&key) {
            Some(entry) if entry.token == token => {
                entry.hits = entry.hits.saturating_add(1);
            }
            Some(entry) => {
                // A different continuation for a key we have seen before. Decay
                // rather than overwrite: a pattern seen many times should
                // survive one deviation, but a stale one must still be able to
                // lose. Overwriting immediately made the cache track the most
                // recent token rather than the most reliable one.
                if entry.hits <= 1 {
                    *entry = Continuation { token, hits: 1 };
                } else {
                    entry.hits -= 1;
                }
            }
            None => {
                if self.table.len() >= self.capacity {
                    self.evict_one();
                }
                self.table.insert(key, Continuation { token, hits: 1 });
            }
        }
    }

    /// Drop a single low-value entry to make room.
    ///
    /// Samples a bounded number of candidates and removes the least-reinforced
    /// rather than scanning the whole table, because eviction sits on the
    /// decode path: an O(n) scan here would cost more than a draft saves.
    fn evict_one(&mut self) {
        const SAMPLE: usize = 8;
        let victim = self
            .table
            .iter()
            .take(SAMPLE)
            .min_by_key(|(_, c)| c.hits)
            .map(|(k, _)| k.clone());
        if let Some(k) = victim {
            self.table.remove(&k);
        }
    }

    /// Propose up to `max_draft` continuation tokens for `history`.
    ///
    /// Prefers the longest matching n-gram, falls back to shorter keys, and
    /// extends the draft by feeding each proposed token back in. Returns empty
    /// when nothing is known — the caller should then decode normally rather
    /// than spend a verify slot on a guess.
    #[must_use]
    pub fn draft(&self, history: &[i32], max_draft: usize) -> Vec<i32> {
        if max_draft == 0 || self.table.is_empty() {
            return Vec::new();
        }
        let mut out: Vec<i32> = Vec::with_capacity(max_draft);
        // Chained proposals run on a local copy of the tail so a draft can
        // build on its own earlier tokens without touching the caller's slice.
        let mut tail: Vec<i32> = history.to_vec();

        for _ in 0..max_draft {
            let Some(next) = self.lookup(&tail) else {
                break;
            };
            // A token that proposes itself is the one degenerate case this can
            // produce unaided; stop rather than spend the verify budget on a
            // run of identical tokens.
            if out.len() >= 2 && out[out.len() - 1] == next && out[out.len() - 2] == next {
                break;
            }
            out.push(next);
            tail.push(next);
        }
        out
    }

    /// Longest-first lookup of the next token for a history tail.
    fn lookup(&self, history: &[i32]) -> Option<i32> {
        for n in (MIN_NGRAM..=MAX_NGRAM).rev() {
            if history.len() < n {
                continue;
            }
            let key = &history[history.len() - n..];
            if let Some(c) = self.table.get(key) {
                return Some(c.token);
            }
        }
        None
    }

    /// Forget everything. For when a cache is reused across tenants and must
    /// not carry one tenant's token history into another's drafts.
    pub fn clear(&mut self) {
        self.table.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_cache_drafts_nothing() {
        let c = NgramCache::default();
        assert!(c.draft(&[1, 2, 3], 4).is_empty());
        assert!(c.is_empty());
    }

    #[test]
    fn a_repeated_sequence_is_drafted_back() {
        let mut c = NgramCache::default();
        c.observe(&[10, 11, 12, 13, 14]);
        // Having seen 12 follow 10,11 we propose it, then chain on.
        assert_eq!(c.draft(&[10, 11], 3), vec![12, 13, 14]);
    }

    #[test]
    fn drafting_stops_at_the_requested_length() {
        let mut c = NgramCache::default();
        c.observe(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(c.draft(&[1, 2], 2).len(), 2);
    }

    #[test]
    fn an_unseen_history_drafts_nothing() {
        let mut c = NgramCache::default();
        c.observe(&[1, 2, 3, 4]);
        assert!(c.draft(&[90, 91], 4).is_empty());
    }

    #[test]
    fn a_longer_match_wins_over_a_shorter_one() {
        let mut c = NgramCache::default();
        // Both histories end in (2,3), but (1,2,3) is the more specific key.
        c.observe(&[1, 2, 3, 100]);
        c.observe(&[9, 9, 2, 3, 200]);
        c.observe(&[9, 9, 2, 3, 200]);
        // (1,2,3) is known only from the first sequence, so it must win here
        // even though (2,3) was reinforced twice toward a different token.
        assert_eq!(c.draft(&[1, 2, 3], 1), vec![100]);
    }

    #[test]
    fn a_reinforced_continuation_survives_one_deviation() {
        let mut c = NgramCache::default();
        for _ in 0..3 {
            c.observe(&[5, 6, 7]);
        }
        // One contradicting observation should decay the count, not replace it.
        c.observe(&[5, 6, 8]);
        assert_eq!(c.draft(&[5, 6], 1), vec![7]);
    }

    #[test]
    fn a_one_off_continuation_is_replaced() {
        let mut c = NgramCache::default();
        c.observe(&[5, 6, 7]);
        c.observe(&[5, 6, 8]);
        assert_eq!(c.draft(&[5, 6], 1), vec![8]);
    }

    #[test]
    fn capacity_is_respected() {
        let mut c = NgramCache::with_capacity(4);
        for i in 0..50i32 {
            c.observe(&[i, i + 1, i + 2, i + 3]);
        }
        assert!(c.len() <= 4, "held {} entries, cap was 4", c.len());
    }

    #[test]
    fn a_self_proposing_token_does_not_run_away() {
        let mut c = NgramCache::default();
        // (7,7) -> 7 is a cycle: unchecked it would fill the whole draft.
        c.observe(&[7, 7, 7, 7, 7, 7]);
        let d = c.draft(&[7, 7], 16);
        assert!(d.len() < 16, "cycle produced a full-length draft: {d:?}");
    }

    #[test]
    fn clearing_forgets_everything() {
        let mut c = NgramCache::default();
        c.observe(&[1, 2, 3, 4]);
        assert!(!c.is_empty());
        c.clear();
        assert!(c.is_empty());
        assert!(c.draft(&[1, 2], 2).is_empty());
    }

    #[test]
    fn a_sequence_shorter_than_the_minimum_order_teaches_nothing() {
        let mut c = NgramCache::default();
        c.observe(&[1, 2]);
        assert!(c.is_empty());
    }
}
