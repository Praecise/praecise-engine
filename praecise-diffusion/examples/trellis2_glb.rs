//! Image to GLB with a TRELLIS.2 checkpoint.
//!
//! ```text
//! cargo run -p praecise-diffusion --release --example trellis2_glb -- \
//!     <checkpoint root> <image.rgba> <width> <height> <out.glb> [seed] [bf16|q8_0|f32]
//! ```
//!
//! The image is raw 8-bit RGBA, top row first, with the object on a
//! transparent background. The checkpoint root holds `pipeline.json`, and the
//! models it names from other repositories under their `owner/name` paths.

use std::path::PathBuf;
use std::time::Instant;

use praecise_diffusion::pipeline::{LoadOptions, Precision};
use praecise_diffusion::trellis2::RgbaImage;
use praecise_diffusion::{Request3d, Trellis2, Trellis2Files, Trellis2Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 5 {
        return Err("usage: trellis2_glb <checkpoint root> <image.rgba> <width> <height> <out.glb> [seed] [bf16|q8_0|f32]".into());
    }
    let (width, height): (u32, u32) = (args[2].parse()?, args[3].parse()?);
    let data = std::fs::read(&args[1])?;
    if data.len() != width as usize * height as usize * 4 {
        return Err(format!("{} holds {} bytes, expected {width}x{height}x4", args[1], data.len()).into());
    }
    let seed = args.get(5).map_or(Ok(0), |s| s.parse())?;
    let precision = match args.get(6).map(String::as_str) {
        None | Some("bf16") => Precision::Bf16,
        Some("q8_0") => Precision::Q8_0,
        Some("f32") => Precision::F32,
        Some(other) => return Err(format!("unknown precision {other}").into()),
    };
    let opts = LoadOptions { precision, cpu_threads: std::thread::available_parallelism().map_or(8, usize::from), device: None };

    let t0 = Instant::now();
    let model = Trellis2::load(&Trellis2Files { root: PathBuf::from(&args[0]) }, opts, Trellis2Path::R512)?;
    eprintln!("loaded in {:.1} s: {model:?}", t0.elapsed().as_secs_f64());

    let asset = model.generate(&Request3d::new(RgbaImage { width, height, data }, seed))?;
    std::fs::write(&args[4], &asset.glb)?;
    eprintln!(
        "wrote {} ({} bytes): {} structure cells, {} surface voxels, raw {:?}, exported {:?}, {} charts, timings {:?}",
        args[4],
        asset.glb.len(),
        asset.structure_cells,
        asset.surface_voxels,
        asset.raw,
        asset.mesh,
        asset.charts,
        asset.timings
    );
    Ok(())
}
