//! Data gates: what may enter a training run, decided before any record reaches a step.
//!
//! - A host policy predicate admits or refuses each record; refusals are counted, never silent.
//! - A record produced by a simulation is admitted only inside the band its simulation was
//!   validated on: every parameter of its origin must have a validated range and lie in it. A
//!   parameter without a range is outside the band.
//! - Evaluation records must be disjoint from the calibration records the simulation was fitted
//!   on and from the training records, compared by content hash.

use std::collections::{BTreeMap, BTreeSet};

use crate::Error;
use crate::hash::Digest;

/// The simulation that produced a record, and the parameter values it ran at.
#[derive(Debug, Clone, PartialEq)]
pub struct SimulationOrigin {
    /// Identifier of the simulation model.
    pub model: String,
    /// Parameter name and value.
    pub parameters: Vec<(String, f64)>,
}

/// The parameter ranges, inclusive, on which a simulation model was validated.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedBand {
    /// Identifier of the simulation model.
    pub model: String,
    /// Parameter name to `(low, high)`.
    pub ranges: BTreeMap<String, (f64, f64)>,
}

impl ValidatedBand {
    /// Whether every parameter of `origin` has a range here and lies inside it.
    #[must_use]
    pub fn contains(&self, origin: &SimulationOrigin) -> bool {
        origin.model == self.model
            && origin.parameters.iter().all(|(name, v)| {
                self.ranges.get(name).is_some_and(|&(lo, hi)| v.is_finite() && lo <= *v && *v <= hi)
            })
    }
}

/// A record a gate can judge.
pub trait GatedRecord {
    /// Content hash of the record.
    fn content_hash(&self) -> Digest;
    /// The simulation that produced it, `None` for a measured record.
    fn simulation(&self) -> Option<&SimulationOrigin>;
}

/// Counts of one gate pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GateReport {
    /// Records admitted.
    pub admitted: usize,
    /// Records the host policy refused.
    pub refused_by_policy: usize,
    /// Simulated records outside every validated band of their model.
    pub refused_outside_band: usize,
}

/// Applies the host policy and the validated bands to `records`, in order. The policy is asked
/// first; a record it refuses is not judged further.
pub fn admit<R: GatedRecord>(
    records: Vec<R>,
    policy: impl Fn(&R) -> bool,
    bands: &[ValidatedBand],
) -> (Vec<R>, GateReport) {
    let mut report = GateReport::default();
    let mut out = Vec::with_capacity(records.len());
    for r in records {
        if !policy(&r) {
            report.refused_by_policy += 1;
        } else if r.simulation().is_some_and(|o| !bands.iter().any(|b| b.contains(o))) {
            report.refused_outside_band += 1;
        } else {
            report.admitted += 1;
            out.push(r);
        }
    }
    (out, report)
}

/// Refuses a run whose evaluation records share content with its calibration or training
/// records.
///
/// # Errors
/// [`Error::Refused`] naming how many evaluation records overlap each set.
pub fn check_disjoint(eval: &[Digest], calibration: &[Digest], train: &[Digest]) -> Result<(), Error> {
    let cal: BTreeSet<&Digest> = calibration.iter().collect();
    let tr: BTreeSet<&Digest> = train.iter().collect();
    let in_cal = eval.iter().filter(|d| cal.contains(d)).count();
    let in_train = eval.iter().filter(|d| tr.contains(d)).count();
    if in_cal + in_train > 0 {
        return Err(Error::Refused(format!(
            "{in_cal} evaluation records appear in the calibration set and {in_train} in the training set"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::sha256;

    struct Rec {
        bytes: Vec<u8>,
        origin: Option<SimulationOrigin>,
    }

    impl GatedRecord for Rec {
        fn content_hash(&self) -> Digest {
            sha256(&self.bytes)
        }
        fn simulation(&self) -> Option<&SimulationOrigin> {
            self.origin.as_ref()
        }
    }

    fn sim(model: &str, params: &[(&str, f64)]) -> Option<SimulationOrigin> {
        Some(SimulationOrigin { model: model.into(), parameters: params.iter().map(|(k, v)| ((*k).into(), *v)).collect() })
    }

    fn band() -> ValidatedBand {
        ValidatedBand { model: "room".into(), ranges: [("rt60".to_string(), (0.2, 0.8)), ("snr".to_string(), (0.0, 30.0))].into() }
    }

    #[test]
    fn simulated_records_are_admitted_only_inside_the_validated_band() {
        let records = vec![
            Rec { bytes: b"measured".to_vec(), origin: None },
            Rec { bytes: b"inside".to_vec(), origin: sim("room", &[("rt60", 0.5), ("snr", 10.0)]) },
            Rec { bytes: b"edge".to_vec(), origin: sim("room", &[("rt60", 0.8)]) },
            Rec { bytes: b"outside".to_vec(), origin: sim("room", &[("rt60", 1.2)]) },
            Rec { bytes: b"unranged".to_vec(), origin: sim("room", &[("rt60", 0.5), ("distance", 2.0)]) },
            Rec { bytes: b"other model".to_vec(), origin: sim("hall", &[("rt60", 0.5)]) },
            Rec { bytes: b"nan".to_vec(), origin: sim("room", &[("rt60", f64::NAN)]) },
        ];
        let (kept, report) = admit(records, |_| true, &[band()]);
        let names: Vec<&[u8]> = kept.iter().map(|r| r.bytes.as_slice()).collect();
        assert_eq!(names, [b"measured".as_slice(), b"inside", b"edge"]);
        assert_eq!(report, GateReport { admitted: 3, refused_by_policy: 0, refused_outside_band: 4 });
    }

    #[test]
    fn the_host_policy_refuses_first_and_is_counted() {
        let records = vec![
            Rec { bytes: b"ok".to_vec(), origin: None },
            Rec { bytes: b"private".to_vec(), origin: None },
            Rec { bytes: b"private sim".to_vec(), origin: sim("room", &[("rt60", 5.0)]) },
        ];
        let (kept, report) = admit(records, |r| !r.bytes.starts_with(b"private"), &[band()]);
        assert_eq!(kept.len(), 1);
        assert_eq!(report, GateReport { admitted: 1, refused_by_policy: 2, refused_outside_band: 0 });
    }

    #[test]
    fn evaluation_overlapping_calibration_or_training_is_refused() {
        let h = |s: &str| sha256(s.as_bytes());
        let eval = [h("e1"), h("e2")];
        assert!(check_disjoint(&eval, &[h("c1")], &[h("t1")]).is_ok());
        let err = check_disjoint(&eval, &[h("e1")], &[h("e2"), h("e1")]).unwrap_err().to_string();
        assert!(err.contains("1 evaluation records appear in the calibration set and 2 in the training set"), "{err}");
    }
}
