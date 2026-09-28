//! Prefix reuse across requests, for every kind of model the batch engine
//! serves.
//!
//! Backend-free: this module decides, the batch engine acts. Everything here is
//! a pure function of what a slot holds, what the new prompt is, and what the
//! loaded model's memory can do, so it is tested without a model.
//!
//! ## What can be rewound depends on the model, not on its name
//!
//! A slot keeps the KV (and any recurrent state) of its last request so the
//! next request can start where the two diverge. Whether the cache can be cut
//! back to that divergence is a property of the model's memory, read from the
//! loaded model through [`MemoryTraits`]:
//!
//! - **Attention** (dense, MoE, a sliding window whose cache keeps every
//!   position): any position can be trimmed. [`Reuse::Trim`].
//! - **A sliding window that forgets**: positions older than the window are
//!   gone, so a trim that needs them cannot be satisfied; a checkpoint of the
//!   window taken earlier can. [`Reuse::Window`].
//! - **Recurrent state** (Mamba, RWKV, gated delta nets, and every hybrid of
//!   those with attention): the state folds in every token, so it can only go
//!   back to a checkpoint. [`Reuse::Recurrent`].
//! - **Bidirectional attention** (embedding and reranking encoders) and
//!   **diffusion decoding**: a prefix's states depend on what follows it, so
//!   nothing carries over. [`Reuse::None`].
//! - **Encoder-decoder**: the decoder's cache hangs off the encoder output, so
//!   what carries over is the encoder output itself when its input repeats.
//!   [`Reuse::EncoderOutput`].
//!
//! ## Where checkpoints go
//!
//! A checkpoint costs memory, so it goes where a following request actually
//! diverges: at the start of a chat turn. An agent's next request repeats the
//! conversation and differs from the previous one at the turn it re-renders,
//! most often the assistant reply that came back without its reasoning or with
//! a stop sequence trimmed. [`TurnEnds`] finds the turn starts in the prompt's
//! own tokens, using the tokens the model's chat template closes a turn with.
//! One more checkpoint, one token before the prompt's end, makes an identical
//! re-send cost a single token.
//!
//! ## Cache namespaces
//!
//! A slot's cache is tagged with the namespace of the request that filled it
//! ([`Namespace`]), derived from the caller's opaque isolation key. A request
//! only ever matches slots in its own namespace, so whether a prompt under
//! another key is resident can neither be reused nor observed through time to
//! first token.

use sha2::{Digest, Sha256};

/// One element of a prompt's identity: a text token, or one position of an
/// image or audio embedding.
///
/// Text tokens are their ids. Media positions carry the top bit and a value
/// derived from the media's content hash and the position's index within it,
/// so two prompts match through a media span only when they carry the same
/// bytes there.
pub type PrefixId = u64;

const MEDIA_BIT: u64 = 1 << 63;

/// Identity of a text token.
pub fn text_id(token: i32) -> PrefixId {
    u64::from(token as u32)
}

/// Identities of the `n` positions a media chunk with this content hash
/// occupies.
pub fn media_ids(content: &[u8; 32], n: usize) -> impl Iterator<Item = PrefixId> + '_ {
    let seed = u64::from_le_bytes(content[..8].try_into().expect("8 bytes"))
        ^ u64::from_le_bytes(content[8..16].try_into().expect("8 bytes"));
    (0..n as u64).map(move |i| MEDIA_BIT | (splitmix64(seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 1))
}

/// Content hash of one media attachment's bytes.
pub fn media_hash(bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"prefix-cache/media");
    h.update(bytes);
    h.finalize().into()
}

/// Where a media chunk sits in a prompt's identity.
///
/// A media chunk is evaluated whole, so a reuse can end before it or after
/// it, never inside it. Its embeddings may also advance the KV position by a
/// different amount than the number of positions it occupies in the identity
/// (multi-axis RoPE gives an image one temporal position per row, not per
/// patch), which is why a cut in the identity is translated to a KV position
/// through [`kv_pos`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaSpan {
    /// Identity index of its first position.
    pub at: usize,
    /// Identity positions it occupies: its token count.
    pub len: usize,
    /// KV positions it advances.
    pub n_pos: usize,
}

/// KV position of identity index `idx` (which must not fall inside a span).
pub fn kv_pos(spans: &[MediaSpan], idx: usize) -> usize {
    spans
        .iter()
        .filter(|s| s.at + s.len <= idx)
        .fold(idx, |pos, s| pos + s.n_pos - s.len)
}

/// The deepest cut at or below `idx` that does not split a media span.
pub fn snap_out_of_media(spans: &[MediaSpan], idx: usize) -> usize {
    spans
        .iter()
        .find(|s| s.at < idx && idx < s.at + s.len)
        .map_or(idx, |s| s.at)
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Whose cache a slot holds. Requests with different namespaces never share
/// a prefix, however similar their prompts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Namespace([u8; 32]);

impl Namespace {
    /// The namespace for a request's cache salt. `None` is the shared
    /// namespace, for callers that set no isolation key.
    pub fn of(salt: Option<&str>) -> Self {
        match salt {
            None => Self::default(),
            Some(s) => {
                let mut h = Sha256::new();
                h.update(b"prefix-cache/namespace");
                h.update((s.len() as u64).to_le_bytes());
                h.update(s.as_bytes());
                Self(h.finalize().into())
            }
        }
    }
}

/// How a loaded model's per-sequence memory behaves, read from the model and
/// the context it is served with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryTraits {
    /// Every layer carries recurrent state.
    pub recurrent: bool,
    /// Attention layers interleaved with recurrent ones.
    pub hybrid: bool,
    /// Sliding attention window in tokens, 0 for none.
    pub n_swa: u32,
    /// The context keeps every position of the sliding-window layers, so they
    /// trim like full attention.
    pub swa_full: bool,
    /// Attention is causal. False for bidirectional encoders.
    pub causal: bool,
    /// The model has an encoder feeding a decoder through cross-attention.
    pub encoder_decoder: bool,
    /// The model decodes by diffusion over a whole canvas.
    pub diffusion: bool,
}

impl Default for MemoryTraits {
    fn default() -> Self {
        Self {
            recurrent: false,
            hybrid: false,
            n_swa: 0,
            swa_full: true,
            causal: true,
            encoder_decoder: false,
            diffusion: false,
        }
    }
}

/// How a slot's cache can be taken back to a shorter prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reuse {
    /// Any position can be trimmed.
    Trim,
    /// Positions older than `n_swa` are dropped as the window moves; going
    /// back past what is still held needs a checkpoint.
    Window { n_swa: u32 },
    /// Recurrent state can only be restored from a checkpoint.
    Recurrent,
    /// Nothing is carried from one request to the next.
    None,
    /// The encoder output is reused when the encoder input repeats.
    EncoderOutput,
}

impl MemoryTraits {
    /// The reuse this memory supports. Recurrent state outranks everything:
    /// a hybrid with a sliding window still cannot trim its recurrent part.
    /// Experts do not appear here at all, because routing does not change what
    /// the cache holds.
    pub fn reuse(&self) -> Reuse {
        if self.diffusion || !self.causal {
            Reuse::None
        } else if self.encoder_decoder {
            Reuse::EncoderOutput
        } else if self.recurrent || self.hybrid {
            Reuse::Recurrent
        } else if self.n_swa > 0 && !self.swa_full {
            Reuse::Window { n_swa: self.n_swa }
        } else {
            Reuse::Trim
        }
    }
}

impl Reuse {
    /// Whether this memory needs checkpoints to rewind at all.
    pub fn needs_checkpoints(self) -> bool {
        matches!(self, Reuse::Window { .. } | Reuse::Recurrent)
    }

    /// Whether a slot's cache survives into the next request.
    pub fn keeps_cache(self) -> bool {
        matches!(self, Reuse::Trim | Reuse::Window { .. } | Reuse::Recurrent)
    }
}

/// How many of a slot's cached positions a new prompt shares, capped so at
/// least the prompt's last element is left to decode (its logits are what the
/// reply is sampled from). A one-element overlap is only the BOS and not
/// worth the bookkeeping.
pub fn shared_prefix(cached: &[PrefixId], prompt: &[PrefixId]) -> usize {
    let common = cached.iter().zip(prompt).take_while(|(a, b)| a == b).count();
    if common < 2 || prompt.is_empty() {
        0
    } else {
        common.min(prompt.len() - 1)
    }
}

/// What to do with a slot's cache before a new prompt is prefilled into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rewind {
    /// The prompt extends everything the slot holds: nothing to drop, prefill
    /// from `.0`.
    Keep(usize),
    /// Drop every position from `.0` on and prefill from there.
    Trim(usize),
    /// Restore the checkpoint taken `.0` positions in, drop attention from
    /// there on, and prefill from there.
    Restore(usize),
    /// Clear the slot and prefill from zero.
    Reset,
}

impl Rewind {
    /// Positions the new request does not have to prefill.
    pub fn reused(self) -> usize {
        match self {
            Rewind::Keep(n) | Rewind::Trim(n) | Rewind::Restore(n) => n,
            Rewind::Reset => 0,
        }
    }
}

/// Decide how to take a slot holding `cached_len` positions back to a shared
/// prefix of `reuse`.
///
/// `checkpoints` are the positions the slot holds checkpoints at, ascending.
/// `pos_min` is the oldest position the slot's cache still holds (for a
/// sliding window, where the window has moved to); it only matters for
/// [`Reuse::Window`]. The deepest usable checkpoint is the one at or below
/// the divergence, and never the prompt's own end, which would leave nothing
/// to decode (`reuse` is already capped below the prompt's length).
pub fn plan_rewind(kind: Reuse, cached_len: usize, reuse: usize, checkpoints: &[usize], pos_min: i64) -> Rewind {
    if reuse == 0 || !kind.keeps_cache() {
        return Rewind::Reset;
    }
    if reuse == cached_len {
        return Rewind::Keep(reuse);
    }
    let checkpoint = || {
        checkpoints
            .iter()
            .rev()
            .find(|&&c| c >= 2 && c <= reuse)
            .map_or(Rewind::Reset, |&c| Rewind::Restore(c))
    };
    match kind {
        Reuse::Trim => Rewind::Trim(reuse),
        // The window before `reuse` must still be held for a trim to leave a
        // correct cache: everything at or after `reuse - n_swa` is attended
        // to by the next token.
        Reuse::Window { n_swa } => {
            if pos_min < reuse as i64 - i64::from(n_swa) {
                Rewind::Trim(reuse)
            } else {
                checkpoint()
            }
        }
        Reuse::Recurrent => checkpoint(),
        Reuse::None | Reuse::EncoderOutput => Rewind::Reset,
    }
}

/// The tokens a model's chat template closes a turn with.
///
/// A turn starts right after one of these. They are read from the template
/// itself (see the engine's probe) and, when there is no template to read,
/// are the vocabulary's end-of-generation tokens, which every template closes
/// an assistant turn with.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TurnEnds(Vec<i32>);

impl TurnEnds {
    pub fn new(mut ids: Vec<i32>) -> Self {
        ids.sort_unstable();
        ids.dedup();
        Self(ids)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Positions where a turn starts in `prompt`: right after each closing
    /// token, strictly inside the prompt.
    pub fn turn_starts(&self, prompt: &[i32]) -> Vec<usize> {
        prompt
            .iter()
            .enumerate()
            .filter(|(i, t)| i + 1 < prompt.len() && self.0.binary_search(t).is_ok())
            .map(|(i, _)| i + 1)
            .collect()
    }
}

/// Where to take checkpoints while prefilling `prompt_len` positions from
/// `from` on.
///
/// Always the last turn start (where the reply being generated now begins, and
/// where the next request re-renders it) and one position before the end (an
/// identical re-send). Earlier turn starts fill what is left of `cap`, latest
/// first, no closer than `min_spacing` to one already chosen: the turns a
/// following request can diverge at are the recent ones.
pub fn checkpoint_cuts(turn_starts: &[usize], from: usize, prompt_len: usize, cap: usize, min_spacing: usize) -> Vec<usize> {
    if cap == 0 || prompt_len < 3 {
        return Vec::new();
    }
    let mut cuts: Vec<usize> = Vec::with_capacity(cap);
    let end = prompt_len - 1;
    if end > from {
        cuts.push(end);
    }
    let mut starts = turn_starts.iter().copied().filter(|&p| p > from && p < end).rev();
    if cuts.len() < cap
        && let Some(last) = starts.next()
    {
        cuts.push(last);
    }
    for p in starts {
        if cuts.len() >= cap {
            break;
        }
        if cuts.iter().all(|&c| c.abs_diff(p) >= min_spacing) {
            cuts.push(p);
        }
    }
    cuts.sort_unstable();
    cuts
}

/// The checkpoints one slot holds, ascending by position, at most `cap`.
#[derive(Debug, Clone)]
pub struct CheckpointStore<S> {
    entries: Vec<(usize, S)>,
}

impl<S> Default for CheckpointStore<S> {
    fn default() -> Self {
        Self { entries: Vec::new() }
    }
}

impl<S> CheckpointStore<S> {
    /// Positions held, ascending.
    pub fn positions(&self) -> Vec<usize> {
        self.entries.iter().map(|(p, _)| *p).collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The checkpoint at exactly `pos`.
    pub fn get(&self, pos: usize) -> Option<&S> {
        self.entries.iter().find(|(p, _)| *p == pos).map(|(_, s)| s)
    }

    /// Keep `state` at `pos`, replacing one already there. Past `cap` the
    /// shallowest goes: a following request diverges late in the
    /// conversation far more often than early.
    pub fn insert(&mut self, pos: usize, state: S, cap: usize) {
        if cap == 0 {
            return;
        }
        match self.entries.binary_search_by_key(&pos, |(p, _)| *p) {
            Ok(i) => self.entries[i].1 = state,
            Err(i) => self.entries.insert(i, (pos, state)),
        }
        while self.entries.len() > cap {
            self.entries.remove(0);
        }
    }

    /// Drop every checkpoint past `pos`: it describes positions the cache no
    /// longer holds, or holds under a different prefix.
    pub fn truncate_after(&mut self, pos: usize) {
        self.entries.retain(|(p, _)| *p <= pos);
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Memory the engine may spend on checkpoints, and what that buys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointBudget {
    /// Bytes available to checkpoints across every slot.
    pub bytes: u64,
    /// Checkpoints each slot may hold.
    pub per_slot: usize,
}

/// Most checkpoints a slot holds however much memory there is.
pub const MAX_CHECKPOINTS_PER_SLOT: usize = 32;

/// The fewest checkpoints worth keeping: the last turn start and the one
/// before the prompt's end.
pub const MIN_USEFUL_CHECKPOINTS: usize = 2;

/// Split `free_bytes` of the memory checkpoints live in between slots.
///
/// Checkpoints are host memory. Half of what is free past a reserve goes to
/// them, the rest is left for everything else the host runs, and the reserve
/// (a tenth of `total_bytes`, at least 2 GiB) is never touched.
pub fn checkpoint_budget(free_bytes: u64, total_bytes: u64, slots: usize, checkpoint_bytes: u64) -> CheckpointBudget {
    let reserve = (total_bytes / 10).max(2 << 30);
    budget_of(free_bytes.saturating_sub(reserve) / 2, slots, checkpoint_bytes)
}

/// Split a fixed `bytes` for checkpoints between slots.
pub fn budget_of(bytes: u64, slots: usize, checkpoint_bytes: u64) -> CheckpointBudget {
    let per_slot = if checkpoint_bytes == 0 || slots == 0 {
        MAX_CHECKPOINTS_PER_SLOT
    } else {
        ((bytes / checkpoint_bytes) / slots as u64).min(MAX_CHECKPOINTS_PER_SLOT as u64) as usize
    };
    CheckpointBudget { bytes, per_slot }
}

/// On a host where the KV cache and checkpoints draw on one memory pool
/// (CPU serving, and unified-memory accelerators), how many tokens of context
/// to give up so every slot can hold the fewest useful checkpoints. `None`
/// when the budget already covers them, or when no context can pay for them
/// without going below `floor_tokens`.
pub fn context_to_release(
    budget: CheckpointBudget,
    slots: usize,
    checkpoint_bytes: u64,
    kv_bytes_per_token: u64,
    n_ctx: u32,
    floor_tokens: u32,
) -> Option<u32> {
    if budget.per_slot >= MIN_USEFUL_CHECKPOINTS || kv_bytes_per_token == 0 {
        return None;
    }
    let needed = checkpoint_bytes
        .saturating_mul((MIN_USEFUL_CHECKPOINTS * slots) as u64)
        .saturating_sub(budget.bytes);
    // Freed KV is split like any free memory: half of it goes to checkpoints.
    let tokens = needed.saturating_mul(2).div_ceil(kv_bytes_per_token);
    let tokens = u32::try_from(tokens).ok()?;
    (n_ctx.checked_sub(tokens)? >= floor_tokens).then_some(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(v: &[i32]) -> Vec<PrefixId> {
        v.iter().map(|t| text_id(*t)).collect()
    }

    fn traits() -> MemoryTraits {
        MemoryTraits::default()
    }

    #[test]
    fn every_memory_kind_is_classified_from_what_the_model_is() {
        assert_eq!(traits().reuse(), Reuse::Trim, "dense attention");
        let moe_dense = traits();
        assert_eq!(moe_dense.reuse(), Reuse::Trim, "experts do not change the cache");
        let swa_full = MemoryTraits { n_swa: 512, ..traits() };
        assert_eq!(swa_full.reuse(), Reuse::Trim, "a window whose cache keeps everything trims");
        let swa = MemoryTraits { n_swa: 512, swa_full: false, ..traits() };
        assert_eq!(swa.reuse(), Reuse::Window { n_swa: 512 });
        let hybrid = MemoryTraits { hybrid: true, n_swa: 4096, swa_full: false, ..traits() };
        assert_eq!(hybrid.reuse(), Reuse::Recurrent, "recurrent state outranks the window");
        let ssm = MemoryTraits { recurrent: true, ..traits() };
        assert_eq!(ssm.reuse(), Reuse::Recurrent);
        let encoder = MemoryTraits { causal: false, ..traits() };
        assert_eq!(encoder.reuse(), Reuse::None, "bidirectional: a prefix depends on its suffix");
        let t5 = MemoryTraits { encoder_decoder: true, ..traits() };
        assert_eq!(t5.reuse(), Reuse::EncoderOutput);
        let diffusion = MemoryTraits { diffusion: true, ..traits() };
        assert_eq!(diffusion.reuse(), Reuse::None);
    }

    #[test]
    fn only_forgetting_memory_needs_checkpoints() {
        assert!(!Reuse::Trim.needs_checkpoints());
        assert!(Reuse::Window { n_swa: 8 }.needs_checkpoints());
        assert!(Reuse::Recurrent.needs_checkpoints());
        assert!(!Reuse::None.needs_checkpoints());
        assert!(!Reuse::None.keeps_cache());
        assert!(!Reuse::EncoderOutput.keeps_cache());
    }

    #[test]
    fn the_shared_prefix_leaves_a_token_to_decode() {
        let same = ids(&[1, 2, 3, 4, 5]);
        assert_eq!(shared_prefix(&same, &same), 4);
        assert_eq!(shared_prefix(&ids(&[1, 77]), &ids(&[1, 90])), 0, "a BOS alone is not reused");
        assert_eq!(shared_prefix(&ids(&[1, 2, 3, 40]), &ids(&[1, 2, 3, 41, 5])), 3);
        assert_eq!(shared_prefix(&[], &ids(&[1, 2])), 0);
    }

    #[test]
    fn an_extension_keeps_everything_on_every_memory_kind() {
        for kind in [Reuse::Trim, Reuse::Window { n_swa: 4 }, Reuse::Recurrent] {
            assert_eq!(plan_rewind(kind, 7, 7, &[], 6), Rewind::Keep(7), "{kind:?}");
        }
    }

    #[test]
    fn attention_trims_to_the_divergence() {
        assert_eq!(plan_rewind(Reuse::Trim, 10, 6, &[], 0), Rewind::Trim(6));
    }

    #[test]
    fn recurrent_state_restores_the_deepest_checkpoint_at_or_below_the_divergence() {
        assert_eq!(plan_rewind(Reuse::Recurrent, 20, 12, &[4, 9, 12, 15], 19), Rewind::Restore(12));
        assert_eq!(plan_rewind(Reuse::Recurrent, 20, 11, &[4, 9, 12, 15], 19), Rewind::Restore(9));
        assert_eq!(plan_rewind(Reuse::Recurrent, 20, 3, &[4, 9], 19), Rewind::Reset, "nothing at or below");
        assert_eq!(plan_rewind(Reuse::Recurrent, 20, 11, &[], 19), Rewind::Reset);
    }

    #[test]
    fn a_window_trims_while_it_still_holds_what_the_next_token_sees() {
        // Cache holds positions 60..100 of a 100-token slot, window 16.
        assert_eq!(plan_rewind(Reuse::Window { n_swa: 16 }, 100, 90, &[], 60), Rewind::Trim(90));
        // Going back to 70 would need 54..70, and 54..60 are gone.
        assert_eq!(plan_rewind(Reuse::Window { n_swa: 16 }, 100, 70, &[40, 65], 60), Rewind::Restore(65));
        assert_eq!(plan_rewind(Reuse::Window { n_swa: 16 }, 100, 70, &[], 60), Rewind::Reset);
    }

    #[test]
    fn nothing_carries_over_without_a_causal_cache() {
        assert_eq!(plan_rewind(Reuse::None, 10, 9, &[5], 0), Rewind::Reset);
        assert_eq!(plan_rewind(Reuse::EncoderOutput, 10, 9, &[5], 0), Rewind::Reset);
        assert_eq!(plan_rewind(Reuse::Trim, 10, 0, &[5], 0), Rewind::Reset);
    }

    #[test]
    fn a_checkpoint_at_the_very_start_is_not_worth_restoring() {
        assert_eq!(plan_rewind(Reuse::Recurrent, 10, 6, &[1], 9), Rewind::Reset);
    }

    #[test]
    fn turn_starts_follow_the_closing_tokens() {
        // 9 closes a turn: [1, u, u, 9, a, a, 9, u, 9, gen, gen]
        let ends = TurnEnds::new(vec![9]);
        let prompt = [1, 5, 5, 9, 6, 6, 9, 5, 9, 7, 7];
        assert_eq!(ends.turn_starts(&prompt), vec![4, 7, 9]);
        // A closing token at the very end starts nothing inside the prompt.
        assert_eq!(ends.turn_starts(&[1, 5, 9]), Vec::<usize>::new());
        assert!(TurnEnds::new(vec![]).turn_starts(&prompt).is_empty());
    }

    #[test]
    fn cuts_keep_the_last_turn_and_the_end_first() {
        let starts = [100, 400, 900, 1500];
        assert_eq!(checkpoint_cuts(&starts, 0, 2000, 2, 256), vec![1500, 1999]);
        assert_eq!(checkpoint_cuts(&starts, 0, 2000, 8, 256), vec![100, 400, 900, 1500, 1999]);
        // Too close to one already chosen: 1450 is within 256 of 1500.
        assert_eq!(checkpoint_cuts(&[900, 1450, 1500], 0, 2000, 8, 256), vec![900, 1500, 1999]);
        // Only positions past the reused prefix are prefilled, so only they
        // can be cut.
        assert_eq!(checkpoint_cuts(&starts, 1000, 2000, 8, 256), vec![1500, 1999]);
        assert!(checkpoint_cuts(&starts, 0, 2000, 0, 256).is_empty());
    }

    #[test]
    fn the_store_keeps_the_deepest_within_its_cap() {
        let mut s = CheckpointStore::default();
        for p in [10, 30, 20, 40] {
            s.insert(p, p * 100, 3);
        }
        assert_eq!(s.positions(), vec![20, 30, 40]);
        s.insert(30, 1, 3);
        assert_eq!(s.get(30), Some(&1));
        s.truncate_after(30);
        assert_eq!(s.positions(), vec![20, 30]);
        s.insert(5, 0, 0);
        assert_eq!(s.positions(), vec![20, 30], "a zero cap keeps nothing new");
    }

    #[test]
    fn media_positions_match_only_the_same_bytes() {
        let a = media_hash(b"image one");
        let b = media_hash(b"image two");
        let ia: Vec<_> = media_ids(&a, 4).collect();
        let ib: Vec<_> = media_ids(&b, 4).collect();
        assert_eq!(ia, media_ids(&a, 4).collect::<Vec<_>>());
        assert!(ia.iter().zip(&ib).all(|(x, y)| x != y));
        assert!(ia.iter().all(|x| x & MEDIA_BIT != 0), "never equal to a text token");
        assert_eq!(ia.len(), 4);
        let unique: std::collections::HashSet<_> = ia.iter().collect();
        assert_eq!(unique.len(), 4, "each position is distinct");
    }

    #[test]
    fn a_media_span_is_reused_whole_or_not_at_all() {
        // Text 0..5, a 16-position image advancing 4 KV positions, text after.
        let spans = [MediaSpan { at: 5, len: 16, n_pos: 4 }];
        assert_eq!(snap_out_of_media(&spans, 3), 3);
        assert_eq!(snap_out_of_media(&spans, 5), 5, "right before the image");
        assert_eq!(snap_out_of_media(&spans, 12), 5, "inside it: back to its start");
        assert_eq!(snap_out_of_media(&spans, 21), 21, "right after it");
        assert_eq!(kv_pos(&spans, 5), 5);
        assert_eq!(kv_pos(&spans, 21), 9, "the image advanced 4 positions, not 16");
        assert_eq!(kv_pos(&spans, 30), 18);
        let plain = [MediaSpan { at: 2, len: 8, n_pos: 8 }];
        assert_eq!(kv_pos(&plain, 20), 20);
    }

    #[test]
    fn namespaces_separate_isolation_keys() {
        assert_eq!(Namespace::of(None), Namespace::default());
        assert_eq!(Namespace::of(Some("alice")), Namespace::of(Some("alice")));
        assert_ne!(Namespace::of(Some("alice")), Namespace::of(Some("bob")));
        assert_ne!(Namespace::of(Some("")), Namespace::of(None));
    }

    #[test]
    fn the_budget_splits_free_memory_between_slots() {
        const GIB: u64 = 1 << 30;
        // GB10-like: 128 GiB total, 60 GiB free after weights and KV.
        let b = checkpoint_budget(60 * GIB, 128 * GIB, 16, 150 << 20);
        assert_eq!(b.bytes, (60 * GIB - 128 * GIB / 10) / 2);
        assert_eq!(b.per_slot, ((b.bytes / (150 << 20)) / 16).min(32) as usize);
        // Nothing free past the reserve buys nothing.
        assert_eq!(checkpoint_budget(GIB, 16 * GIB, 4, 1 << 20).per_slot, 0);
        // A tiny state is capped, not unbounded.
        assert_eq!(checkpoint_budget(60 * GIB, 128 * GIB, 4, 1024).per_slot, MAX_CHECKPOINTS_PER_SLOT);
    }

    #[test]
    fn context_is_released_only_when_it_buys_the_minimum() {
        const MIB: u64 = 1 << 20;
        let short = CheckpointBudget { bytes: 100 * MIB, per_slot: 0 };
        // 4 slots x 2 checkpoints x 50 MiB = 400 MiB, 300 MiB short; freed KV
        // is halved, so 600 MiB of it at 128 KiB a token = 4800 tokens.
        assert_eq!(context_to_release(short, 4, 50 * MIB, 128 << 10, 32_768, 8_192), Some(4_800));
        // Not below the floor.
        assert_eq!(context_to_release(short, 4, 50 * MIB, 128 << 10, 12_000, 8_192), None);
        let enough = CheckpointBudget { bytes: 1 << 40, per_slot: 4 };
        assert_eq!(context_to_release(enough, 4, 50 * MIB, 128 << 10, 32_768, 8_192), None);
    }
}
