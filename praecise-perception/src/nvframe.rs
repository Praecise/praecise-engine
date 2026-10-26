//! NVIDIA Metropolis `nv.Frame` messages (schema.proto, package `nv`), the
//! per-frame object message Metropolis analytics and Halos Safety Core read.
//!
//! Only the fields a tracker fills are written: the frame's version, id,
//! timestamp and sensor; per object its id, box, type, confidence, info,
//! speed, direction and, when the camera has a ground-plane calibration, its
//! position on the ground. The encoding is plain protobuf wire format, so no
//! generated code or protobuf library is needed.

/// One object of a frame.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Object {
    /// `Object.id`: the track id.
    pub id: String,
    /// `Object.bbox`: left, top, right, bottom in pixels.
    pub bbox: [f32; 4],
    /// `Object.type`, for instance `Person` or `Vehicle`.
    pub kind: String,
    /// `Object.confidence`.
    pub confidence: f32,
    /// `Object.info` entries.
    pub info: Vec<(String, String)>,
    /// `Object.speed`, in metres per second on the ground plane when
    /// `coordinate` is set, otherwise pixels per second.
    pub speed: f32,
    /// `Object.dir`: the unit direction of motion.
    pub dir: Option<[f32; 2]>,
    /// `Object.coordinate`: position on the ground plane in metres.
    pub coordinate: Option<[f64; 3]>,
}

/// One frame.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Frame {
    /// `Frame.version`.
    pub version: String,
    /// `Frame.id`.
    pub id: String,
    /// `Frame.timestamp`, seconds and nanoseconds since the Unix epoch.
    pub timestamp: (i64, i32),
    /// `Frame.sensorId`.
    pub sensor_id: String,
    /// `Frame.objects`.
    pub objects: Vec<Object>,
    /// `Frame.info` entries.
    pub info: Vec<(String, String)>,
}

fn varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn key(out: &mut Vec<u8>, field: u32, wire: u32) {
    varint(out, u64::from(field << 3 | wire));
}

fn string(out: &mut Vec<u8>, field: u32, s: &str) {
    if !s.is_empty() {
        key(out, field, 2);
        varint(out, s.len() as u64);
        out.extend_from_slice(s.as_bytes());
    }
}

fn message(out: &mut Vec<u8>, field: u32, body: &[u8]) {
    key(out, field, 2);
    varint(out, body.len() as u64);
    out.extend_from_slice(body);
}

fn float(out: &mut Vec<u8>, field: u32, v: f32) {
    if v != 0.0 {
        key(out, field, 5);
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn double(out: &mut Vec<u8>, field: u32, v: f64) {
    if v != 0.0 {
        key(out, field, 1);
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn map_entries(out: &mut Vec<u8>, field: u32, entries: &[(String, String)]) {
    for (k, v) in entries {
        let mut e = Vec::new();
        string(&mut e, 1, k);
        string(&mut e, 2, v);
        message(out, field, &e);
    }
}

impl Object {
    fn encode(&self) -> Vec<u8> {
        let mut o = Vec::new();
        string(&mut o, 1, &self.id);
        let mut b = Vec::new();
        float(&mut b, 1, self.bbox[0]);
        float(&mut b, 2, self.bbox[1]);
        float(&mut b, 3, self.bbox[2]);
        float(&mut b, 4, self.bbox[3]);
        float(&mut b, 6, self.confidence);
        message(&mut o, 2, &b);
        string(&mut o, 3, &self.kind);
        float(&mut o, 4, self.confidence);
        map_entries(&mut o, 5, &self.info);
        float(&mut o, 10, self.speed);
        if let Some(d) = self.dir {
            // packed repeated float
            key(&mut o, 11, 2);
            varint(&mut o, 8);
            o.extend_from_slice(&d[0].to_le_bytes());
            o.extend_from_slice(&d[1].to_le_bytes());
        }
        if let Some(c) = self.coordinate {
            let mut m = Vec::new();
            double(&mut m, 1, c[0]);
            double(&mut m, 2, c[1]);
            double(&mut m, 3, c[2]);
            message(&mut o, 12, &m);
        }
        o
    }
}

impl Frame {
    /// The message in protobuf wire format.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut f = Vec::new();
        string(&mut f, 1, &self.version);
        string(&mut f, 2, &self.id);
        let mut ts = Vec::new();
        if self.timestamp.0 != 0 {
            key(&mut ts, 1, 0);
            varint(&mut ts, self.timestamp.0 as u64);
        }
        if self.timestamp.1 != 0 {
            key(&mut ts, 2, 0);
            varint(&mut ts, i64::from(self.timestamp.1) as u64);
        }
        message(&mut f, 3, &ts);
        string(&mut f, 4, &self.sensor_id);
        for o in &self.objects {
            message(&mut f, 5, &o.encode());
        }
        map_entries(&mut f, 12, &self.info);
        f
    }

    /// Append the message to `out` prefixed by its length as a varint, the
    /// delimited form a stream of messages is written in.
    pub fn write_delimited(&self, out: &mut Vec<u8>) {
        let body = self.encode();
        varint(out, body.len() as u64);
        out.extend_from_slice(&body);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_encodes_to_the_expected_bytes() {
        let f = Frame {
            version: "4.0".into(),
            id: "7".into(),
            timestamp: (1, 500),
            sensor_id: "cam".into(),
            objects: vec![Object { id: "3".into(), bbox: [1.0, 2.0, 3.0, 4.0], kind: "Person".into(), confidence: 0.5, ..Object::default() }],
            info: Vec::new(),
        };
        let b = f.encode();
        // version, id, timestamp {1: 1, 2: 500}, sensorId
        assert_eq!(&b[..15], &[0x0a, 3, b'4', b'.', b'0', 0x12, 1, b'7', 0x1a, 5, 0x08, 1, 0x10, 0xf4, 0x03]);
        assert_eq!(&b[15..20], &[0x22, 3, b'c', b'a', b'm']);
        // objects: field 5, then id "3" and the bbox message (4 floats and confidence, 25 bytes)
        assert_eq!(b[20], 0x2a);
        assert_eq!(&b[22..27], &[0x0a, 1, b'3', 0x12, 25]);
        assert_eq!(&b[27..32], &[0x0d, 0, 0, 0x80, 0x3f]);
        let mut d = Vec::new();
        f.write_delimited(&mut d);
        assert_eq!(usize::from(d[0]), b.len());
        assert_eq!(&d[1..], &b[..]);
    }

    #[test]
    fn varints_span_bytes() {
        let mut v = Vec::new();
        varint(&mut v, 300);
        assert_eq!(v, vec![0xac, 0x02]);
    }
}
