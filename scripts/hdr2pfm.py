#!/usr/bin/env python3
# Converts a Radiance RGBE (.hdr) image to a binary little-endian PFM, which is the
# only environment map format the renderer reads. ImageMagick builds without HDRI
# support clamp everything above 1.0 on the way, so the sun disappears; this keeps
# the full float range. Needs numpy only.
#
# Usage:
#   ./scripts/hdr2pfm.py sky.hdr sky.pfm

import sys
import numpy as np


def read_hdr(path):
    with open(path, "rb") as f:
        data = f.read()

    if not data.startswith(b"#?"):
        raise ValueError("not a Radiance HDR file")

    # header lines end at the first empty line, the resolution string follows
    position = 0
    while True:
        end = data.index(b"\n", position)
        line = data[position:end]
        position = end + 1
        if line == b"":
            break
    end = data.index(b"\n", position)
    resolution = data[position:end].split()
    position = end + 1
    if resolution[0] != b"-Y" or resolution[2] != b"+X":
        raise ValueError("only the standard -Y h +X w orientation is supported")
    height = int(resolution[1])
    width = int(resolution[3])

    raw = np.frombuffer(data, dtype=np.uint8, offset=position)
    rgbe = np.empty((height, width, 4), dtype=np.uint8)
    cursor = 0
    for row in range(height):
        # new style run length encoding: 2, 2, width_hi, width_lo, then 4 planes of runs
        if width >= 8 and width < 32768 and raw[cursor] == 2 and raw[cursor + 1] == 2 and (int(raw[cursor + 2]) << 8 | int(raw[cursor + 3])) == width:
            cursor += 4
            for channel in range(4):
                column = 0
                while column < width:
                    count = int(raw[cursor]); cursor += 1
                    if count > 128:
                        count -= 128
                        rgbe[row, column:column + count, channel] = raw[cursor]; cursor += 1
                    else:
                        rgbe[row, column:column + count, channel] = raw[cursor:cursor + count]; cursor += count
                    column += count
        else:
            rgbe[row] = raw[cursor:cursor + width * 4].reshape(width, 4)
            cursor += width * 4

    exponent = rgbe[:, :, 3].astype(np.int32)
    scale = np.where(exponent > 0, np.ldexp(1.0, exponent - 136), 0.0).astype(np.float32)
    rgb = rgbe[:, :, :3].astype(np.float32) * scale[:, :, None]
    return rgb


def write_pfm(path, rgb):
    height, width, _ = rgb.shape
    with open(path, "wb") as f:
        f.write(b"PF\n%d %d\n-1.0\n" % (width, height))
        # PFM stores the bottom row first
        f.write(np.ascontiguousarray(rgb[::-1], dtype="<f4").tobytes())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: hdr2pfm.py input.hdr output.pfm")
    image = read_hdr(sys.argv[1])
    write_pfm(sys.argv[2], image)
    print("%s: %dx%d, max %.1f -> %s" % (sys.argv[1], image.shape[1], image.shape[0], float(image.max()), sys.argv[2]))
