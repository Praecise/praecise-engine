//! Parameter-delta and optimizer-state primitives.
//!
//! Narrow, deterministic functions over a model's trainable state: export the
//! parameter delta since a named checkpoint, import a delta, apply a delta with
//! a weight, and export or import optimizer state. Each output is content
//! hashed. How deltas or optimizer state move between machines is outside the
//! engine; these functions only produce and consume bytes.

use std::collections::BTreeMap;

use crate::Error;
use crate::canonical::{Decoder, Encoder};
use crate::checkpoint::{DType, Tensor, TrainState};
use crate::hash::{Digest, domain_hash};

/// Name prefix of trainable parameters in a [`TrainState`].
pub const PARAM_PREFIX: &str = "param.";
/// Name prefix of optimizer state in a [`TrainState`].
pub const OPT_PREFIX: &str = "opt.";

/// `current - since` for every F32 parameter, tagged with the root of `since`.
#[derive(Debug, Clone, PartialEq)]
pub struct Delta {
    /// State root the delta is relative to.
    pub since: Digest,
    /// Per-parameter differences, by full tensor name.
    pub tensors: BTreeMap<String, (Vec<u64>, Vec<f32>)>,
}

impl Delta {
    /// Canonical encoding.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut e = Encoder::new();
        e.str("praecise.delta.v1")
            .digest(&self.since)
            .u64(self.tensors.len() as u64);
        for (name, (shape, values)) in &self.tensors {
            e.str(name).u64(shape.len() as u64);
            for &d in shape {
                e.u64(d);
            }
            e.u64(values.len() as u64);
            for &v in values {
                e.f32(v);
            }
        }
        e.finish()
    }

    /// Content hash of the canonical encoding.
    #[must_use]
    pub fn root(&self) -> Digest {
        domain_hash("praecise.delta", &[&self.encode()])
    }

    /// Parses an encoded delta and checks it against an expected content hash.
    ///
    /// # Errors
    /// [`Error::Format`] on malformed bytes, [`Error::Mismatch`] when the bytes
    /// do not hash to `expected`.
    pub fn import(bytes: &[u8], expected: &Digest) -> Result<Self, Error> {
        let got = domain_hash("praecise.delta", &[bytes]);
        if got != *expected {
            return Err(Error::Mismatch(format!(
                "delta hashes to {got}, expected {expected}"
            )));
        }
        let mut d = Decoder::new(bytes);
        if d.str()? != "praecise.delta.v1" {
            return Err(Error::Format("not a delta".into()));
        }
        let since = d.digest()?;
        let mut tensors = BTreeMap::new();
        for _ in 0..d.u64()? {
            let name = d.str()?;
            let rank = d.u64()?;
            let shape = (0..rank).map(|_| d.u64()).collect::<Result<Vec<_>, _>>()?;
            let n = d.u64()?;
            if Some(n) != shape.iter().try_fold(1u64, |a, &b| a.checked_mul(b)) {
                return Err(Error::Format(format!(
                    "delta {name}: element count does not match shape"
                )));
            }
            let values = (0..n).map(|_| d.f32()).collect::<Result<Vec<_>, _>>()?;
            tensors.insert(name, (shape, values));
        }
        d.finish()?;
        Ok(Self { since, tensors })
    }
}

fn params(state: &TrainState) -> impl Iterator<Item = (&String, &Tensor)> {
    state
        .iter()
        .filter(|(n, t)| n.starts_with(PARAM_PREFIX) && t.dtype == DType::F32)
}

/// The delta of `current` relative to `since`: the same F32 parameters must
/// exist in both with the same shapes.
///
/// # Errors
/// [`Error::Mismatch`] when the parameter sets or shapes differ.
pub fn export_delta(current: &TrainState, since: &TrainState) -> Result<Delta, Error> {
    let mut tensors = BTreeMap::new();
    for (name, cur) in params(current) {
        let base = since.get(name).ok_or_else(|| {
            Error::Mismatch(format!("parameter {name} is missing from the base state"))
        })?;
        if base.shape != cur.shape || base.dtype != DType::F32 {
            return Err(Error::Mismatch(format!(
                "parameter {name} changed shape or type"
            )));
        }
        let (c, b) = (cur.to_f32()?, base.to_f32()?);
        tensors.insert(
            name.clone(),
            (
                cur.shape.clone(),
                c.iter().zip(&b).map(|(x, y)| x - y).collect(),
            ),
        );
    }
    if params(since).count() != tensors.len() {
        return Err(Error::Mismatch(
            "the base state has parameters the current state lacks".into(),
        ));
    }
    Ok(Delta {
        since: since.state_root(),
        tensors,
    })
}

/// `state + weight * delta` for every parameter in the delta, element by
/// element in index order; tensors the delta does not name are copied.
///
/// # Errors
/// [`Error::Mismatch`] when a named parameter is missing or has another shape.
pub fn apply_delta(state: &TrainState, delta: &Delta, weight: f32) -> Result<TrainState, Error> {
    let mut out = TrainState::new();
    for (name, t) in state.iter() {
        let new = match delta.tensors.get(name) {
            Some((shape, d)) => {
                if *shape != t.shape {
                    return Err(Error::Mismatch(format!(
                        "delta {name} has shape {shape:?}, state has {:?}",
                        t.shape
                    )));
                }
                let v = t.to_f32()?;
                let w: Vec<f32> = v
                    .iter()
                    .zip(d)
                    .map(|(x, dx)| weight.mul_add(*dx, *x))
                    .collect();
                Tensor::from_f32(shape.clone(), &w)
            }
            None => t.clone(),
        };
        out.insert(name.clone(), new)?;
    }
    for name in delta.tensors.keys() {
        if state.get(name).is_none() {
            return Err(Error::Mismatch(format!(
                "delta names {name}, which the state lacks"
            )));
        }
    }
    Ok(out)
}

/// Optimizer state: every `opt.*` tensor and the optimizer step count.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizerState {
    /// Number of optimizer steps taken.
    pub step: u64,
    /// The optimizer tensors.
    pub tensors: TrainState,
}

impl OptimizerState {
    /// Extracts the optimizer tensors of a state.
    ///
    /// # Errors
    /// Never in practice; kept fallible for tensor validation.
    pub fn export(state: &TrainState, step: u64) -> Result<Self, Error> {
        let mut tensors = TrainState::new();
        for (name, t) in state.iter().filter(|(n, _)| n.starts_with(OPT_PREFIX)) {
            tensors.insert(name.clone(), t.clone())?;
        }
        Ok(Self { step, tensors })
    }

    /// Canonical encoding.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut e = Encoder::new();
        e.str("praecise.optstate.v1")
            .u64(self.step)
            .bytes(&self.tensors.payload());
        e.finish()
    }

    /// Content hash of the canonical encoding.
    #[must_use]
    pub fn root(&self) -> Digest {
        domain_hash("praecise.optstate", &[&self.encode()])
    }

    /// Parses an encoded optimizer state against an expected content hash.
    ///
    /// # Errors
    /// [`Error::Mismatch`] when the hash differs, [`Error::Format`] on bad bytes.
    pub fn import(bytes: &[u8], expected: &Digest) -> Result<Self, Error> {
        let got = domain_hash("praecise.optstate", &[bytes]);
        if got != *expected {
            return Err(Error::Mismatch(format!(
                "optimizer state hashes to {got}, expected {expected}"
            )));
        }
        let mut d = Decoder::new(bytes);
        if d.str()? != "praecise.optstate.v1" {
            return Err(Error::Format("not an optimizer state".into()));
        }
        let step = d.u64()?;
        let tensors = TrainState::from_payload(d.bytes()?)?;
        d.finish()?;
        Ok(Self { step, tensors })
    }

    /// Replaces the optimizer tensors of `state` with these, keeping parameters.
    ///
    /// # Errors
    /// [`Error::Mismatch`] when a tensor's shape differs from the one it replaces.
    pub fn install(&self, state: &TrainState) -> Result<TrainState, Error> {
        let mut out = TrainState::new();
        for (name, t) in state.iter().filter(|(n, _)| !n.starts_with(OPT_PREFIX)) {
            out.insert(name.clone(), t.clone())?;
        }
        for (name, t) in self.tensors.iter() {
            if let Some(old) = state.get(name)
                && (old.shape != t.shape || old.dtype != t.dtype)
            {
                return Err(Error::Mismatch(format!(
                    "optimizer tensor {name} changed shape or type"
                )));
            }
            out.insert(name.clone(), t.clone())?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(a: &[f32], m: f32) -> TrainState {
        let mut s = TrainState::new();
        s.insert("param.w", Tensor::from_f32(vec![a.len() as u64], a))
            .unwrap();
        s.insert(
            "opt.m.param.w",
            Tensor::from_f32(vec![a.len() as u64], &vec![m; a.len()]),
        )
        .unwrap();
        s
    }

    #[test]
    fn export_apply_round_trip() {
        let base = state(&[1.0, 2.0, 3.0], 0.0);
        let cur = state(&[1.5, 1.0, 3.0], 0.1);
        let delta = export_delta(&cur, &base).unwrap();
        assert_eq!(delta.since, base.state_root());
        assert_eq!(delta.tensors["param.w"].1, vec![0.5, -1.0, 0.0]);
        let back = apply_delta(&base, &delta, 1.0).unwrap();
        assert_eq!(back.get("param.w"), cur.get("param.w"));
        let half = apply_delta(&base, &delta, 0.5).unwrap();
        assert_eq!(
            half.get("param.w").unwrap().to_f32().unwrap(),
            vec![1.25, 1.5, 3.0]
        );
        // optimizer tensors are not part of a delta
        assert!(!delta.tensors.contains_key("opt.m.param.w"));
    }

    #[test]
    fn import_checks_hash_and_shapes() {
        let delta = export_delta(&state(&[2.0, 0.0], 0.0), &state(&[1.0, 1.0], 0.0)).unwrap();
        let bytes = delta.encode();
        assert_eq!(Delta::import(&bytes, &delta.root()).unwrap(), delta);
        assert!(matches!(
            Delta::import(&bytes, &crate::hash::sha256(b"x")),
            Err(Error::Mismatch(_))
        ));
        assert!(apply_delta(&state(&[1.0, 2.0, 3.0], 0.0), &delta, 1.0).is_err());
        assert!(export_delta(&state(&[1.0], 0.0), &state(&[1.0, 2.0], 0.0)).is_err());
    }

    #[test]
    fn optimizer_state_round_trip() {
        let s = state(&[1.0, 2.0], 0.25);
        let opt = OptimizerState::export(&s, 17).unwrap();
        assert_eq!(opt.tensors.len(), 1);
        let bytes = opt.encode();
        let back = OptimizerState::import(&bytes, &opt.root()).unwrap();
        assert_eq!(back, opt);
        let fresh = state(&[1.0, 2.0], 0.0);
        let installed = back.install(&fresh).unwrap();
        assert_eq!(installed.get("opt.m.param.w"), s.get("opt.m.param.w"));
        assert_eq!(installed.get("param.w"), fresh.get("param.w"));
    }
}
