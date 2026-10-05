//! Pictures and video frames as model input, for the scoring and embedding
//! entry points that read them through a vision projector.

/// A picture: packed RGB rows, already sized the way the model's own
/// processor sizes it (the projector keeps a picture whose sides are
/// multiples of its patch grid and whose area is within its limits).
#[derive(Debug, Clone)]
pub struct Picture {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width * height * 3` bytes.
    pub rgb: Vec<u8>,
    /// A frame of a video. Consecutive frames whose markers follow each
    /// other are merged over time by projectors with a temporal patch.
    pub video_frame: bool,
}

/// Text that may carry pictures and video: each media marker
/// ([`llama_cpp_2::mtmd::mtmd_default_marker`]) stands for the next entry of
/// `pictures`, in order.
#[derive(Debug, Clone, Default)]
pub struct MediaInput {
    /// The text, with one marker per picture.
    pub text: String,
    /// The pictures, in marker order.
    pub pictures: Vec<Picture>,
}

impl MediaInput {
    /// Check that the markers and the pictures agree.
    ///
    /// # Errors
    /// When the counts differ.
    pub fn check(&self, what: &str) -> crate::error::Result<()> {
        let markers = self.text.matches(llama_cpp_2::mtmd::mtmd_default_marker()).count();
        if markers == self.pictures.len() {
            Ok(())
        } else {
            Err(crate::error::Error::Inference(format!("{what} has {markers} media markers for {} pictures", self.pictures.len())))
        }
    }

    /// The pictures as projector bitmaps, video frames marked mergeable.
    pub(crate) fn bitmaps(&self) -> crate::error::Result<Vec<llama_cpp_2::mtmd::MtmdBitmap>> {
        self.pictures
            .iter()
            .map(|p| {
                let mut bitmap = llama_cpp_2::mtmd::MtmdBitmap::from_image_data(p.width, p.height, &p.rgb)
                    .map_err(|e| crate::error::Error::Inference(format!("a picture could not be read: {e}")))?;
                bitmap.set_mergeable(p.video_frame);
                Ok(bitmap)
            })
            .collect()
    }
}
