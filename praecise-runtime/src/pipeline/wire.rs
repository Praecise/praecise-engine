//! Messages carried inside the encrypted frames of a pipeline link.

use super::PipelineError;

/// Ticket value that addresses every request in flight.
pub const ALL_TICKETS: u64 = u64::MAX;

/// Hidden states for a micro-batch, travelling down the chain.
#[derive(Debug, Clone, PartialEq)]
pub struct Forward {
    /// Request this micro-batch belongs to.
    pub ticket: u64,
    /// Position of each row in its sequence.
    pub positions: Vec<i32>,
    /// Sequence of each row.
    pub seqs: Vec<i32>,
    /// Rows whose logits the last stage returns (0 or 1 per row).
    pub outputs: Vec<u8>,
    /// Floats per row.
    pub width: u32,
    /// Residual stream, `positions.len() * width` floats, row-major.
    pub hidden: Vec<f32>,
}

/// One message on a link.
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    /// Hidden states, downstream.
    Forward(Forward),
    /// Logits of the output rows of a ticket, upstream; `rows * width` floats.
    Logits {
        /// Request answered.
        ticket: u64,
        /// Number of output rows.
        rows: u32,
        /// Floats per row (the vocabulary size).
        width: u32,
        /// The logits, row-major.
        data: Vec<f32>,
    },
    /// A stage failed a ticket (or [`ALL_TICKETS`]), upstream.
    Failure {
        /// Request that failed, or [`ALL_TICKETS`].
        ticket: u64,
        /// Index of the stage that failed.
        stage: u32,
        /// What happened.
        reason: String,
    },
    /// Drop positions `from_pos..` of a sequence from every stage's cache (`from_pos`
    /// -1 drops the whole sequence), downstream.
    Truncate {
        /// Sequence.
        seq: i32,
        /// First position dropped.
        from_pos: i32,
    },
}

const T_FORWARD: u8 = 1;
const T_LOGITS: u8 = 2;
const T_FAILURE: u8 = 3;
const T_TRUNCATE: u8 = 4;

fn put_f32s(out: &mut Vec<u8>, v: &[f32]) {
    out.reserve(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

fn put_i32s(out: &mut Vec<u8>, v: &[i32]) {
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

impl Message {
    /// Serialize.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Message::Forward(f) => {
                out.reserve(32 + f.positions.len() * 9 + f.hidden.len() * 4);
                out.push(T_FORWARD);
                out.extend_from_slice(&f.ticket.to_le_bytes());
                out.extend_from_slice(&(f.positions.len() as u32).to_le_bytes());
                out.extend_from_slice(&f.width.to_le_bytes());
                put_i32s(&mut out, &f.positions);
                put_i32s(&mut out, &f.seqs);
                out.extend_from_slice(&f.outputs);
                put_f32s(&mut out, &f.hidden);
            }
            Message::Logits { ticket, rows, width, data } => {
                out.push(T_LOGITS);
                out.extend_from_slice(&ticket.to_le_bytes());
                out.extend_from_slice(&rows.to_le_bytes());
                out.extend_from_slice(&width.to_le_bytes());
                put_f32s(&mut out, data);
            }
            Message::Failure { ticket, stage, reason } => {
                out.push(T_FAILURE);
                out.extend_from_slice(&ticket.to_le_bytes());
                out.extend_from_slice(&stage.to_le_bytes());
                out.extend_from_slice(reason.as_bytes());
            }
            Message::Truncate { seq, from_pos } => {
                out.push(T_TRUNCATE);
                out.extend_from_slice(&seq.to_le_bytes());
                out.extend_from_slice(&from_pos.to_le_bytes());
            }
        }
        out
    }

    /// Parse.
    ///
    /// # Errors
    ///
    /// [`PipelineError::Protocol`] for a malformed message.
    pub fn decode(bytes: &[u8]) -> Result<Self, PipelineError> {
        let mut r = Cursor { b: bytes, at: 0 };
        let msg = match r.u8()? {
            T_FORWARD => {
                let ticket = r.u64()?;
                let n = r.u32()? as usize;
                let width = r.u32()?;
                let positions = r.i32s(n)?;
                let seqs = r.i32s(n)?;
                let outputs = r.take(n)?.to_vec();
                let hidden = r.f32s(n.checked_mul(width as usize).ok_or_else(|| proto("size"))?)?;
                Message::Forward(Forward { ticket, positions, seqs, outputs, width, hidden })
            }
            T_LOGITS => {
                let ticket = r.u64()?;
                let rows = r.u32()?;
                let width = r.u32()?;
                let data = r.f32s((rows as usize).checked_mul(width as usize).ok_or_else(|| proto("size"))?)?;
                Message::Logits { ticket, rows, width, data }
            }
            T_FAILURE => {
                let ticket = r.u64()?;
                let stage = r.u32()?;
                let reason = String::from_utf8_lossy(r.take(bytes.len() - r.at)?).into_owned();
                Message::Failure { ticket, stage, reason }
            }
            T_TRUNCATE => Message::Truncate { seq: r.i32()?, from_pos: r.i32()? },
            t => return Err(proto(&format!("unknown message type {t}"))),
        };
        if r.at != bytes.len() {
            return Err(proto("trailing bytes"));
        }
        Ok(msg)
    }
}

fn proto(what: &str) -> PipelineError {
    PipelineError::Protocol(format!("malformed message: {what}"))
}

struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], PipelineError> {
        let end = self.at.checked_add(n).filter(|e| *e <= self.b.len()).ok_or_else(|| proto("truncated"))?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, PipelineError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, PipelineError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().map_err(|_| proto("u32"))?))
    }
    fn i32(&mut self) -> Result<i32, PipelineError> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().map_err(|_| proto("i32"))?))
    }
    fn u64(&mut self) -> Result<u64, PipelineError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().map_err(|_| proto("u64"))?))
    }
    fn i32s(&mut self, n: usize) -> Result<Vec<i32>, PipelineError> {
        let b = self.take(n.checked_mul(4).ok_or_else(|| proto("size"))?)?;
        Ok(b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, PipelineError> {
        let b = self.take(n.checked_mul(4).ok_or_else(|| proto("size"))?)?;
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_round_trip_bit_exact_and_reject_truncation() {
        let msgs = vec![
            Message::Forward(Forward {
                ticket: 7,
                positions: vec![0, 1, 2],
                seqs: vec![0, 0, 1],
                outputs: vec![0, 0, 1],
                width: 2,
                hidden: vec![1.5, -0.0, f32::MIN_POSITIVE, 3.25e-30, 1e30, -7.0],
            }),
            Message::Logits { ticket: 7, rows: 1, width: 3, data: vec![0.1, 0.2, 0.3] },
            Message::Failure { ticket: ALL_TICKETS, stage: 2, reason: "gone".into() },
            Message::Truncate { seq: 3, from_pos: -1 },
        ];
        for m in msgs {
            let enc = m.encode();
            let dec = Message::decode(&enc).unwrap();
            assert_eq!(dec.encode(), enc);
            if !matches!(m, Message::Failure { .. }) {
                assert!(Message::decode(&enc[..enc.len() - 1]).is_err());
            }
        }
        assert!(Message::decode(&[99]).is_err());
    }
}
