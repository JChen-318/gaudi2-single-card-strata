"""Independent float32/IEEE-half oracle for original --mtp-q4 row encoding."""
import math
import pathlib
import random
import struct
import subprocess
import sys
import tempfile


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def encode(values):
    out = bytearray()
    for base in range(0, len(values), 32):
        row = values[base:base + 32]
        maximum = signed = 0.
        for value in row:
            if maximum < abs(value):
                maximum, signed = abs(value), value
        scale = f32(signed / -8.)
        inverse = f32(1. / scale) if scale else 0.
        out += struct.pack('<e', scale)
        codes = [min(15, math.trunc(f32(f32(value * inverse) + 8.5))) for value in row]
        out += bytes(codes[j] | codes[j + 16] << 4 for j in range(16))
    return out


def main(executable):
    rng = random.Random(3181221)
    cases = []
    for columns, rows in [(32, 1), (64, 3), (640, 1), (2560, 3), (65536, 1), (32, 128)]:
        for corpus in range(3):
            values = [0.] * (columns * rows)
            if corpus == 1:
                values = [f32(rng.uniform(-30, 30)) for _ in values]
            elif corpus == 2:
                values = [f32((i % 17 - 8) * .5) for i in range(len(values))]
                for base in range(0, len(values), 32):
                    values[base:base + 2] = [8., -8.] if base % 64 else [-8., 8.]
            cases.append((columns, values, encode(values)))
    with tempfile.TemporaryDirectory(prefix='strata-hpu-q4-oracle-') as directory:
        oracle = pathlib.Path(directory) / 'oracle.bin'
        with oracle.open('wb') as out:
            out.write(struct.pack('<I', len(cases)))
            for columns, values, expected in cases:
                out.write(struct.pack('<II', columns, len(values)))
                out.write(struct.pack(f'<{len(values)}f', *values))
                out.write(expected)
        subprocess.run([str(executable), str(oracle)], check=True)


if __name__ == '__main__':
    main(sys.argv[1])
