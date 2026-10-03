//! The step contract and the hash-linked step log.
//!
//! A step is a pure function `step(StepSpec) -> StepResult`. Each record's hash
//! is `H(prev_record_hash || StepSpec || StepResult)`, so the head of the log
//! commits to the whole training history; a verifier that recomputes any step
//! and gets a different result can name the first record that disagrees.

use std::fs;
use std::io::Write as _;
use std::path::Path;

use crate::Error;
use crate::canonical::{Decoder, Encoder};
use crate::hash::{Digest, domain_hash};

/// Everything a step is a function of.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepSpec {
    /// Content root of the frozen base weights.
    pub base_root: Digest,
    /// Root of the trainable state the step starts from.
    pub state_root: Digest,
    /// Root of the data shard batches are drawn from.
    pub data_root: Digest,
    /// Seed of the batch selection and of every random draw in the step.
    pub sample_seed: u64,
    /// Position of the step in the run.
    pub step_index: u64,
    /// Hash of the canonical recipe.
    pub recipe_hash: Digest,
    /// Kernel class identifier the step runs on.
    pub kernel_class: String,
}

impl StepSpec {
    /// Canonical encoding.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut e = Encoder::new();
        e.digest(&self.base_root)
            .digest(&self.state_root)
            .digest(&self.data_root)
            .u64(self.sample_seed)
            .u64(self.step_index)
            .digest(&self.recipe_hash)
            .str(&self.kernel_class);
        e.finish()
    }

    fn decode(d: &mut Decoder<'_>) -> Result<Self, Error> {
        Ok(Self {
            base_root: d.digest()?,
            state_root: d.digest()?,
            data_root: d.digest()?,
            sample_seed: d.u64()?,
            step_index: d.u64()?,
            recipe_hash: d.digest()?,
            kernel_class: d.str()?,
        })
    }
}

/// What a step produced.
#[derive(Debug, Clone, PartialEq)]
pub struct StepResult {
    /// Root of the trainable state after the step.
    pub state_root: Digest,
    /// Objective value.
    pub loss: f64,
    /// Global gradient norm before any clipping.
    pub grad_norm: f64,
    /// Digest of probe outputs (for example per-token log-probs of a fixed prompt set).
    pub probe_digest: Digest,
    /// Digest of the rewards the step consumed, when the objective uses rewards.
    pub reward_digest: Option<Digest>,
}

impl StepResult {
    /// Canonical encoding.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut e = Encoder::new();
        e.digest(&self.state_root)
            .f64(self.loss)
            .f64(self.grad_norm)
            .digest(&self.probe_digest)
            .opt_digest(self.reward_digest.as_ref());
        e.finish()
    }

    fn decode(d: &mut Decoder<'_>) -> Result<Self, Error> {
        Ok(Self {
            state_root: d.digest()?,
            loss: d.f64()?,
            grad_norm: d.f64()?,
            probe_digest: d.digest()?,
            reward_digest: d.opt_digest()?,
        })
    }

    /// Whether two results are bitwise identical (floats compared by bit pattern).
    #[must_use]
    pub fn bitwise_eq(&self, other: &Self) -> bool {
        self.encode() == other.encode()
    }
}

/// One link of the log.
#[derive(Debug, Clone, PartialEq)]
pub struct StepRecord {
    /// Hash of the previous record ([`Digest::ZERO`] for the first).
    pub prev: Digest,
    /// The step's inputs.
    pub spec: StepSpec,
    /// The step's outputs.
    pub result: StepResult,
    /// `H(prev || spec || result)`.
    pub hash: Digest,
}

/// Hash of a record.
#[must_use]
pub fn record_hash(prev: &Digest, spec: &StepSpec, result: &StepResult) -> Digest {
    domain_hash(
        "praecise.step",
        &[&prev.0, &spec.encode(), &result.encode()],
    )
}

/// An append-only, hash-linked log of steps.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StepLog {
    records: Vec<StepRecord>,
}

impl StepLog {
    /// An empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Hash of the last record, or [`Digest::ZERO`] when empty.
    #[must_use]
    pub fn head(&self) -> Digest {
        self.records.last().map_or(Digest::ZERO, |r| r.hash)
    }

    /// Records in order.
    #[must_use]
    pub fn records(&self) -> &[StepRecord] {
        &self.records
    }

    /// Appends a step. A step must start from the previous step's resulting
    /// state, on the same base and recipe, at the next index.
    ///
    /// # Errors
    /// [`Error::Mismatch`] when the step does not continue the log.
    pub fn append(&mut self, spec: StepSpec, result: StepResult) -> Result<&StepRecord, Error> {
        if let Some(last) = self.records.last() {
            if spec.state_root != last.result.state_root {
                return Err(Error::Mismatch(format!(
                    "step {} starts from {} but step {} ended at {}",
                    spec.step_index, spec.state_root, last.spec.step_index, last.result.state_root
                )));
            }
            if spec.step_index != last.spec.step_index + 1 {
                return Err(Error::Mismatch(format!(
                    "step index {} does not follow {}",
                    spec.step_index, last.spec.step_index
                )));
            }
            if spec.base_root != last.spec.base_root || spec.recipe_hash != last.spec.recipe_hash {
                return Err(Error::Mismatch(
                    "base or recipe changed within one log".into(),
                ));
            }
        }
        let prev = self.head();
        let hash = record_hash(&prev, &spec, &result);
        self.records.push(StepRecord {
            prev,
            spec,
            result,
            hash,
        });
        Ok(&self.records[self.records.len() - 1])
    }

    /// Checks every link and every continuity rule.
    ///
    /// # Errors
    /// [`Error::Mismatch`] naming the first bad record.
    pub fn verify(&self) -> Result<(), Error> {
        let mut rebuilt = Self::new();
        for (i, r) in self.records.iter().enumerate() {
            let got = rebuilt
                .append(r.spec.clone(), r.result.clone())
                .map_err(|e| Error::Mismatch(format!("record {i}: {e}")))?;
            if got.hash != r.hash || got.prev != r.prev {
                return Err(Error::Mismatch(format!("record {i}: hash link broken")));
            }
        }
        Ok(())
    }

    /// The index of the first record whose result differs from `replayed`
    /// (results compared bitwise), or `None` when all agree.
    #[must_use]
    pub fn first_divergence(&self, replayed: &[StepResult]) -> Option<usize> {
        self.records
            .iter()
            .zip(replayed)
            .position(|(r, s)| !r.result.bitwise_eq(s))
            .or_else(|| (replayed.len() < self.records.len()).then_some(replayed.len()))
    }

    /// Canonical encoding of the whole log.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut e = Encoder::new();
        e.str("praecise.steplog.v1").u64(self.records.len() as u64);
        for r in &self.records {
            e.bytes(&r.spec.encode())
                .bytes(&r.result.encode())
                .digest(&r.hash);
        }
        e.finish()
    }

    /// Parses and verifies an encoded log.
    ///
    /// # Errors
    /// [`Error::Format`] on malformed bytes, [`Error::Mismatch`] on a broken link.
    pub fn decode(bytes: &[u8]) -> Result<Self, Error> {
        let mut d = Decoder::new(bytes);
        if d.str()? != "praecise.steplog.v1" {
            return Err(Error::Format("not a step log".into()));
        }
        let n = d.u64()?;
        let mut log = Self::new();
        for i in 0..n {
            let mut sd = Decoder::new(d.bytes()?);
            let spec = StepSpec::decode(&mut sd)?;
            sd.finish()?;
            let mut rd = Decoder::new(d.bytes()?);
            let result = StepResult::decode(&mut rd)?;
            rd.finish()?;
            let hash = d.digest()?;
            let got = log.append(spec, result)?.hash;
            if got != hash {
                return Err(Error::Mismatch(format!(
                    "record {i}: stored hash does not match"
                )));
            }
        }
        d.finish()?;
        Ok(log)
    }

    /// Writes the log to `path` (atomically).
    ///
    /// # Errors
    /// [`Error::Io`] on write failure.
    pub fn save(&self, path: &Path) -> Result<(), Error> {
        let tmp = path.with_extension("tmp");
        let mut f = fs::File::create(&tmp)?;
        f.write_all(&self.encode())?;
        f.sync_all()?;
        fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Reads and verifies a log from `path`.
    ///
    /// # Errors
    /// [`Error::Io`], [`Error::Format`] or [`Error::Mismatch`].
    pub fn load(path: &Path) -> Result<Self, Error> {
        Self::decode(&fs::read(path)?)
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use crate::hash::sha256;

    fn spec(i: u64, state: Digest) -> StepSpec {
        StepSpec {
            base_root: sha256(b"base"),
            state_root: state,
            data_root: sha256(b"data"),
            sample_seed: 9,
            step_index: i,
            recipe_hash: sha256(b"recipe"),
            kernel_class: "cpu/x86_64/ks1".into(),
        }
    }

    fn result(i: u64) -> StepResult {
        StepResult {
            state_root: sha256(&i.to_le_bytes()),
            loss: 1.0 / (i as f64 + 1.0),
            grad_norm: 0.5,
            probe_digest: sha256(b"probe"),
            reward_digest: None,
        }
    }

    fn log(n: u64) -> StepLog {
        let mut l = StepLog::new();
        let mut state = sha256(b"init");
        for i in 0..n {
            let r = result(i);
            let next = r.state_root;
            l.append(spec(i, state), r).unwrap();
            state = next;
        }
        l
    }

    #[test]
    fn log_round_trip_and_head_commits() {
        let l = log(5);
        l.verify().unwrap();
        let d = StepLog::decode(&l.encode()).unwrap();
        assert_eq!(d, l);
        let mut other = log(4);
        let mut r = result(4);
        r.loss = 0.123;
        other.append(spec(4, result(3).state_root), r).unwrap();
        assert_ne!(other.head(), l.head());
    }

    #[test]
    fn discontinuities_are_refused() {
        let mut l = log(2);
        assert!(l.append(spec(2, sha256(b"elsewhere")), result(2)).is_err());
        assert!(l.append(spec(5, result(1).state_root), result(5)).is_err());
        let mut s = spec(2, result(1).state_root);
        s.recipe_hash = sha256(b"other");
        assert!(l.append(s, result(2)).is_err());
    }

    #[test]
    fn tampering_and_divergence_are_found() {
        let l = log(4);
        let mut bytes = l.encode();
        let n = bytes.len();
        bytes[n - 40] ^= 1;
        assert!(StepLog::decode(&bytes).is_err());
        let mut replay: Vec<StepResult> = (0..4).map(result).collect();
        assert_eq!(l.first_divergence(&replay), None);
        replay[2].grad_norm = f64::from_bits(replay[2].grad_norm.to_bits() + 1);
        assert_eq!(l.first_divergence(&replay), Some(2));
        assert_eq!(l.first_divergence(&replay[..1]), Some(1));
    }
}
