//! Binary glTF 2.0 (`.glb`) export of one textured triangle mesh with a
//! metallic-roughness PBR material, byte-for-byte deterministic for the same
//! input (fixed JSON key order, fixed PNG encoding).

use std::fmt::Write as _;
use std::io::Write as _;

use crate::error::{Error, Result};

/// An RGBA or RGB 8-bit image, row-major, top row first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Texture {
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// 3 (RGB) or 4 (RGBA).
    pub channels: u8,
    /// `width * height * channels` bytes.
    pub data: Vec<u8>,
}

/// A triangle mesh ready for export, in glTF's frame (Y up, metres).
#[derive(Debug, Clone, Default)]
pub struct TexturedMesh {
    /// Vertex positions.
    pub positions: Vec<[f32; 3]>,
    /// Unit vertex normals.
    pub normals: Vec<[f32; 3]>,
    /// Texture coordinates (origin at the image's top-left corner).
    pub uvs: Vec<[f32; 2]>,
    /// Counter-clockwise triangles.
    pub indices: Vec<[u32; 3]>,
    /// Base colour (RGB) and alpha.
    pub base_color: Option<Texture>,
    /// Occlusion (R, unused), roughness (G), metallic (B).
    pub metallic_roughness: Option<Texture>,
    /// Render both faces.
    pub double_sided: bool,
}

const GLB_MAGIC: u32 = 0x4654_6C67;
const CHUNK_JSON: u32 = 0x4E4F_534A;
const CHUNK_BIN: u32 = 0x004E_4942;

/// Encode `mesh` as a GLB file.
///
/// # Errors
/// [`Error::Request`] for an empty or inconsistent mesh.
pub fn write_glb(mesh: &TexturedMesh) -> Result<Vec<u8>> {
    let nv = mesh.positions.len();
    if nv == 0 || mesh.indices.is_empty() {
        return Err(Error::Request("an empty mesh cannot be exported".into()));
    }
    if mesh.normals.len() != nv || mesh.uvs.len() != nv {
        return Err(Error::Request("normals and texture coordinates must match the vertices".into()));
    }
    if mesh.indices.iter().flatten().any(|&i| i as usize >= nv) {
        return Err(Error::Request("a triangle refers to a missing vertex".into()));
    }
    let mut bin: Vec<u8> = Vec::new();
    let mut views: Vec<(usize, usize, Option<u32>)> = Vec::new();
    let mut push_view = |bin: &mut Vec<u8>, bytes: &[u8], target: Option<u32>| -> usize {
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        let off = bin.len();
        bin.extend_from_slice(bytes);
        views.push((off, bytes.len(), target));
        views.len() - 1
    };
    let f32s = |v: &mut Vec<u8>, x: f32| v.extend_from_slice(&x.to_le_bytes());
    let mut pos = Vec::with_capacity(nv * 12);
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for p in &mesh.positions {
        for a in 0..3 {
            if !p[a].is_finite() {
                return Err(Error::Request("a vertex position is not finite".into()));
            }
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
            f32s(&mut pos, p[a]);
        }
    }
    let mut nrm = Vec::with_capacity(nv * 12);
    for n in &mesh.normals {
        n.iter().for_each(|&x| f32s(&mut nrm, x));
    }
    let mut uv = Vec::with_capacity(nv * 8);
    for t in &mesh.uvs {
        t.iter().for_each(|&x| f32s(&mut uv, x));
    }
    let mut idx = Vec::with_capacity(mesh.indices.len() * 12);
    for t in &mesh.indices {
        t.iter().for_each(|&i| idx.extend_from_slice(&i.to_le_bytes()));
    }
    let v_pos = push_view(&mut bin, &pos, Some(34962));
    let v_nrm = push_view(&mut bin, &nrm, Some(34962));
    let v_uv = push_view(&mut bin, &uv, Some(34962));
    let v_idx = push_view(&mut bin, &idx, Some(34963));
    let mut images = Vec::new();
    for tex in [&mesh.base_color, &mesh.metallic_roughness].into_iter().flatten() {
        let png = encode_png(tex)?;
        images.push(push_view(&mut bin, &png, None));
    }
    while bin.len() % 4 != 0 {
        bin.push(0);
    }

    let mut j = String::new();
    j.push_str(r#"{"asset":{"version":"2.0","generator":"Praecise Engine"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],"#);
    let material = mesh.base_color.is_some() || mesh.metallic_roughness.is_some();
    let _ = write!(
        j,
        r#""meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2}},"indices":3,"mode":4{}}}]}}],"#,
        if material { r#","material":0"# } else { "" }
    );
    if material {
        let mut pbr = String::from(r#""baseColorFactor":[1,1,1,1],"metallicFactor":1,"roughnessFactor":1"#);
        let mut t = 0;
        if mesh.base_color.is_some() {
            let _ = write!(pbr, r#","baseColorTexture":{{"index":{t}}}"#);
            t += 1;
        }
        if mesh.metallic_roughness.is_some() {
            let _ = write!(pbr, r#","metallicRoughnessTexture":{{"index":{t}}}"#);
        }
        let _ = write!(
            j,
            r#""materials":[{{"pbrMetallicRoughness":{{{pbr}}},"alphaMode":"OPAQUE","doubleSided":{}}}],"#,
            mesh.double_sided
        );
        j.push_str(r#""samplers":[{"magFilter":9729,"minFilter":9729,"wrapS":33071,"wrapT":33071}],"#);
        let tex: Vec<String> = (0..images.len()).map(|i| format!(r#"{{"sampler":0,"source":{i}}}"#)).collect();
        let _ = write!(j, r#""textures":[{}],"#, tex.join(","));
        let img: Vec<String> = images.iter().map(|v| format!(r#"{{"bufferView":{v},"mimeType":"image/png"}}"#)).collect();
        let _ = write!(j, r#""images":[{}],"#, img.join(","));
    }
    let num = |x: f32| format!("{x:?}");
    let _ = write!(
        j,
        r#""accessors":[{{"bufferView":{v_pos},"componentType":5126,"count":{nv},"type":"VEC3","min":[{},{},{}],"max":[{},{},{}]}},{{"bufferView":{v_nrm},"componentType":5126,"count":{nv},"type":"VEC3"}},{{"bufferView":{v_uv},"componentType":5126,"count":{nv},"type":"VEC2"}},{{"bufferView":{v_idx},"componentType":5125,"count":{},"type":"SCALAR"}}],"#,
        num(lo[0]),
        num(lo[1]),
        num(lo[2]),
        num(hi[0]),
        num(hi[1]),
        num(hi[2]),
        mesh.indices.len() * 3
    );
    let bv: Vec<String> = views
        .iter()
        .map(|(o, l, t)| match t {
            Some(t) => format!(r#"{{"buffer":0,"byteOffset":{o},"byteLength":{l},"target":{t}}}"#),
            None => format!(r#"{{"buffer":0,"byteOffset":{o},"byteLength":{l}}}"#),
        })
        .collect();
    let _ = write!(j, r#""bufferViews":[{}],"buffers":[{{"byteLength":{}}}]}}"#, bv.join(","), bin.len());
    let mut jb = j.into_bytes();
    while jb.len() % 4 != 0 {
        jb.push(b' ');
    }
    let total = 12 + 8 + jb.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&GLB_MAGIC.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(jb.len() as u32).to_le_bytes());
    out.extend_from_slice(&CHUNK_JSON.to_le_bytes());
    out.extend_from_slice(&jb);
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(&CHUNK_BIN.to_le_bytes());
    out.extend_from_slice(&bin);
    Ok(out)
}

fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, e) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *e = c;
    }
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c = table[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

fn png_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let mut c = kind.to_vec();
    c.extend_from_slice(data);
    out.extend_from_slice(&c);
    out.extend_from_slice(&crc32(&c).to_be_bytes());
}

/// Encode a texture as a PNG (no filtering, zlib level 6: deterministic).
///
/// # Errors
/// [`Error::Request`] for a mis-sized texture.
pub fn encode_png(t: &Texture) -> Result<Vec<u8>> {
    let ch = usize::from(t.channels);
    if !(ch == 3 || ch == 4) || t.data.len() != t.width as usize * t.height as usize * ch || t.width == 0 || t.height == 0 {
        return Err(Error::Request("texture size does not match its data".into()));
    }
    let mut raw = Vec::with_capacity(t.data.len() + t.height as usize);
    let row = t.width as usize * ch;
    for y in 0..t.height as usize {
        raw.push(0u8);
        raw.extend_from_slice(&t.data[y * row..(y + 1) * row]);
    }
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(6));
    z.write_all(&raw).map_err(Error::Io)?;
    let comp = z.finish().map_err(Error::Io)?;
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&t.width.to_be_bytes());
    ihdr.extend_from_slice(&t.height.to_be_bytes());
    ihdr.extend_from_slice(&[8, if ch == 4 { 6 } else { 2 }, 0, 0, 0]);
    png_chunk(&mut out, b"IHDR", &ihdr);
    png_chunk(&mut out, b"IDAT", &comp);
    png_chunk(&mut out, b"IEND", &[]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad() -> TexturedMesh {
        TexturedMesh {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            uvs: vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            indices: vec![[0, 1, 2], [0, 2, 3]],
            base_color: Some(Texture { width: 2, height: 2, channels: 4, data: vec![255; 16] }),
            metallic_roughness: Some(Texture { width: 2, height: 2, channels: 3, data: vec![0, 128, 0, 0, 128, 0, 0, 128, 0, 0, 128, 0] }),
            double_sided: true,
        }
    }

    #[test]
    fn glb_layout_is_well_formed_and_deterministic() {
        let a = write_glb(&quad()).unwrap();
        let b = write_glb(&quad()).unwrap();
        assert_eq!(a, b);
        assert_eq!(&a[0..4], b"glTF");
        assert_eq!(u32::from_le_bytes(a[8..12].try_into().unwrap()) as usize, a.len());
        let jl = u32::from_le_bytes(a[12..16].try_into().unwrap()) as usize;
        assert_eq!(jl % 4, 0);
        let json: serde_json::Value = serde_json::from_slice(&a[20..20 + jl]).unwrap();
        assert_eq!(json["accessors"][0]["max"][1], serde_json::json!(1.0));
        assert_eq!(json["images"].as_array().unwrap().len(), 2);
        let bl = u32::from_le_bytes(a[20 + jl..24 + jl].try_into().unwrap()) as usize;
        assert_eq!(json["buffers"][0]["byteLength"].as_u64().unwrap() as usize, bl);
    }

    #[test]
    fn png_crc_matches_known_value() {
        assert_eq!(crc32(b"IEND"), 0xAE42_6082);
    }
}
