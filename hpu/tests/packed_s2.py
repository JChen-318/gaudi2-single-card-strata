"""Independent float32 reciprocal/multiply/signed-half/trunc oracle for routed ActQ."""
import pathlib
import random
import struct
import subprocess
import sys
import tempfile
from packed_quant import f32


def oracle(values):
    codes, metadata = bytearray(), bytearray()
    for base in range(0, len(values), 32):
        block = values[base:base+32]
        maximum = max(map(abs, block))
        scale = f32(maximum / 127) if maximum else 0.0
        inverse = f32(1 / scale) if scale else 0.0
        quantized = []
        for value in block:
            product = f32(value * inverse)
            rounded = f32(product + (0.5 if product >= 0 else -0.5))
            quantized.append(int(min(127, max(-127, rounded))))
        codes += bytes(code & 255 for code in quantized)
        metadata += struct.pack('<ff', scale, f32(scale * sum(quantized)))
    return codes, metadata


def main(executable):
    rng = random.Random(448)
    cases = [(32, [0.0] * 32), (32, [127.0, 0.5, -0.5, 1.5, -1.5] + [0.4] * 27),
             (64, [1e8] * 64), (64, [(-1 if i % 2 else 1) * 1e20 for i in range(64)]),
             (64, [1e-6 * (i-30) for i in range(64)])]
    for columns in (32, 64, 128, 640, 2560, 65536):
        for rows in (1, 3):
            cases.append((columns, [rng.uniform(-7, 7) for _ in range(columns * rows)]))
    cases.append((64, [rng.uniform(-1, 1) for _ in range(64 * 128)]))
    with tempfile.TemporaryDirectory(prefix='strata-hpu-s2-') as directory:
        path = pathlib.Path(directory) / 's2_oracle.bin'
        with path.open('wb') as stream:
            stream.write(struct.pack('<I', len(cases)))
            for columns, data in cases:
                data = list(map(f32, data))
                codes, metadata = oracle(data)
                stream.write(struct.pack('<II', columns, len(data)))
                stream.write(struct.pack('<' + 'f' * len(data), *data))
                stream.write(codes)
                stream.write(metadata)
        subprocess.run([executable, str(path)], check=True)


if __name__ == '__main__':
    main(sys.argv[1])
