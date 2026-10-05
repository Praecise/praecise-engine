#!/bin/sh
# Regenerate the decode fixtures: the test pattern of tests/roundtrip.rs
# (64x48, 12 frames, 24 fps) written by other encoders, BT.709 studio range,
# with B-frames where the format has them. Needs python3 and an ffmpeg build
# with libx264, libx265 and libaom; the fixtures are data only, nothing here
# is linked into the crate.
set -eu
cd "$(dirname "$0")"
FFMPEG=${FFMPEG:-ffmpeg}
python3 - <<'PY' > pattern.rgb
import math, sys
w, h, n = 64, 48, 12
out = bytearray()
for i in range(n):
    for y in range(h):
        for x in range(w):
            out.append(round(128 + 100 * math.sin(x / 9 + i * 0.5)))
            out.append(round(128 + 100 * math.sin(y / 7 - i * 0.3)))
            out.append(round(128 + 100 * math.sin((x + y) / 11 + i * 0.8)))
sys.stdout.buffer.write(out)
PY
IN="-f rawvideo -pix_fmt rgb24 -s 64x48 -r 24 -i pattern.rgb"
COLOR="-vf scale=out_color_matrix=bt709:out_range=tv -pix_fmt yuv420p -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv"
$FFMPEG -y -v error $IN $COLOR -c:v libx264 -profile:v high -bf 2 -crf 14 h264-high-bframes.mp4
$FFMPEG -y -v error $IN $COLOR -c:v libx264 -profile:v main -bf 2 -crf 14 h264.mkv
$FFMPEG -y -v error $IN $COLOR -c:v libx265 -tag:v hvc1 -x265-params bframes=2:log-level=error -crf 14 h265-main-bframes.mp4
$FFMPEG -y -v error $IN $COLOR -c:v libx265 -x265-params bframes=2:log-level=error -crf 14 h265.mkv
$FFMPEG -y -v error $IN $COLOR -c:v libaom-av1 -crf 20 -b:v 0 -cpu-used 8 av1.webm
$FFMPEG -y -v error $IN $COLOR -c:v libaom-av1 -crf 20 -b:v 0 -cpu-used 8 av1.mp4
rm pattern.rgb
ls -l *.mp4 *.mkv *.webm
