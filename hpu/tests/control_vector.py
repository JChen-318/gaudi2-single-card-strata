"""Original controlvector GGUF fixtures; all tests are host-only."""
import pathlib
import struct
import subprocess
import sys
import tempfile


def text(value):
    encoded = value.encode()
    return struct.pack('<Q', len(encoded)) + encoded


def fixture(path, hidden=64, rows=None, architecture='controlvector', kind=0):
    if rows is None:
        rows = [('direction.0', [99.] * hidden), ('direction.1', [3., 4.] + [0.] * (hidden - 2)),
                ('direction.2', [0., 0., 2.] + [0.] * (hidden - 3)),
                ('direction.3', [0., 0., 0., 1.] + [0.] * (hidden - 4)),
                ('direction.100', [99.] * hidden)]
    header = bytearray(struct.pack('<IIQQ', 0x46554747, 3, len(rows), 1))
    header += text('general.architecture') + struct.pack('<I', 8) + text(architecture)
    payload = bytearray()
    for name, values in rows:
        header += text(name) + struct.pack('<IQIQ', 1, len(values), kind, len(payload))
        payload += struct.pack('<' + 'f' * len(values), *values)
        payload += bytes((-len(payload)) % 32)
    header += bytes((-len(header)) % 32)
    path.write_bytes(header + payload)


def main(executable):
    with tempfile.TemporaryDirectory(prefix='strata-hpu-control-vector-') as directory:
        root = pathlib.Path(directory)
        fixture(root / 'a.gguf')
        fixture(root / 'b.gguf', rows=[('direction.1', [1., -2.] + [0.] * 62), ('direction.2', [0., 0., 1.] + [0.] * 61)])
        fixture(root / 'wrong_arch.gguf', architecture='qwen4exp')
        fixture(root / 'wrong_type.gguf', kind=30)
        fixture(root / 'wrong_shape.gguf', hidden=63)
        fixture(root / 'nan.gguf', rows=[('direction.1', [float('nan')] + [0.] * 63)])
        fixture(root / 'inf.gguf', rows=[('direction.1', [float('inf')] + [0.] * 63)])
        fixture(root / 'duplicate.gguf', rows=[('direction.1', [0.] * 64)] * 2)
        fixture(root / 'malformed.gguf', rows=[('direction.1junk', [0.] * 64)])
        fixture(root / 'zero.gguf', rows=[('direction.1', [0.] * 64)])
        fixture(root / 'overflow.gguf', rows=[('direction.1', [3e38, 3e38] + [0.] * 62)])
        fixture(root / 'truncated.gguf', rows=[('direction.1', [0.] * 64)])
        data = (root / 'truncated.gguf').read_bytes()
        (root / 'truncated.gguf').write_bytes(data[:-1])
        subprocess.run([executable, str(root)], check=True)


if __name__ == '__main__':
    main(sys.argv[1])
