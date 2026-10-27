"""Independent Python/struct IEEE-half oracle for production host Q8_1 staging."""
import math
import pathlib
import random
import struct
import subprocess
import sys
import tempfile


def f32(value):
    try:
        return struct.unpack('<f', struct.pack('<f', value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def oracle(values):
    codes, metadata = bytearray(), bytearray()
    for base in range(0, len(values), 32):
        x = values[base:base+32]
        sums = x[:]
        for offset in (16, 8, 4, 2, 1):
            sums = [f32(sums[i] + sums[i ^ offset]) for i in range(32)]
        maximum = max(map(abs, x))
        scale = min(f32(maximum / 127), 65504)
        assert not maximum or scale
        for value in x:
            q = f32(value / scale) if scale else 0
            q = min(127, max(-127, q))
            rounded = math.copysign(math.floor(abs(q) + 0.5), q)
            codes.append(int(rounded) & 255)
        for value in (scale, sums[0]):
            assert not math.isnan(value)
            half = struct.pack('<e', min(65504, max(-65504, value)))
            metadata += struct.pack('<f', struct.unpack('<e', half)[0])
    return codes, metadata


def main(executable):
    randomizer = random.Random(605)
    cases = [(32, [0.0] * 32), (32, [127.0, 0.5, -0.5, 1.5, -1.5] + [0.4] * 27),
             (64, [1e8] * 64), (64, [(-1 if i % 2 else 1) * 1e20 for i in range(64)]),
             (64, [1e-6 * (i-30) for i in range(64)])]
    for columns in (32, 64, 128, 640, 2560, 65536):
        for rows in (1, 3):
            cases.append((columns, [randomizer.uniform(-7, 7) for _ in range(columns * rows)]))
    cases.append((64, [randomizer.uniform(-1, 1) for _ in range(64 * 128)]))
    with tempfile.TemporaryDirectory(prefix='strata-hpu-packed-') as directory:
        path = pathlib.Path(directory) / 'q8_1_oracle.bin'
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
