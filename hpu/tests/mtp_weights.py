"""No SDK/device: original MTP runtime sparse files and malformed catalogs."""
import pathlib
import struct
import subprocess
import sys
import tempfile

H, FF, NE = 2560, 640, 512
BLOB = 3 * H * FF * 18 // 64


def fixture(root):
    records = []

    def tensor(name, kind, rows, cols):
        offset = (records[-1][4] + records[-1][5] + 255) // 256 * 256 if records else 0
        size = rows * cols // 32 * 34 if kind == 'q8_0' else rows * cols * (4 if kind == 'f32' else 2)
        records.append((name, kind, rows, cols, offset, size))

    for name in ['fc_embedding.weight', 'fc_hidden.weight']:
        tensor(name, 'q8_0', H, H)
    for name, rows in [('q', 24 * 512), ('k', 512), ('v', 512)]:
        tensor(f'self_attn.{name}_proj.weight', 'q8_0', rows, H)
    tensor('self_attn.o_proj.weight', 'q8_0', H, 24 * 256)
    tensor('pre_fc_norm_embedding.weight', 'f32', 1, H)
    tensor('pre_fc_norm_hidden.weight', 'f32', 1, 4 * H)
    for name in ['q', 'k']:
        tensor(f'self_attn.{name}_norm.weight', 'f32', 1, 256)
    for prefix in ['attn_hyper_connection.', 'mlp_hyper_connection.', 'hyper_connection_mixer.']:
        tensor(prefix + 'hc_norm.weight', 'f32', 1, 4 * H)
        tensor(prefix + 'input_mix_weight_down.weight', 'bf16', 320, 4 * H)
        tensor(prefix + 'input_mix_weight_up.weight', 'bf16', 4 * H, 320)
        if prefix != 'hyper_connection_mixer.':
            tensor(prefix + 'block_inject_weight.weight', 'bf16', 4, 4 * H)
    tensor('mlp.gate.weight', 'bf16', NE, H)
    tensor('mlp.shared_expert_gate.weight', 'bf16', 1, H)
    for name in ['gate', 'up']:
        tensor(f'mlp.shared_expert.{name}_proj.weight', 'q8_0', FF, H)
    tensor('mlp.shared_expert.down_proj.weight', 'q8_0', H, FF)
    lines = [' '.join(map(str, record)) for record in records]
    index = root / 'dense.txt'
    index.write_text('\n'.join(lines) + '\n')
    with (root / 'dense.bin').open('wb') as file:
        file.seek(records[-1][4] + records[-1][5] - 1)
        file.write(b'\0')
        file.seek(0)
        file.write(struct.pack('<e32b', 1, *range(-16, 16)))
        norm = next(record for record in records if record[0] == 'pre_fc_norm_embedding.weight')
        file.seek(norm[4])
        file.write(struct.pack('<2f', 1.25, -0.5))
    with (root / 'experts.bin').open('wb') as file:
        file.seek(NE * BLOB - 1)
        file.write(b'\0')
        base = 511 * BLOB
        gu_codes, d_codes = 0, 2 * FF * H // 4
        gu_scales = d_codes + H * FF // 4
        d_scales = gu_scales + 2 * FF * (H // 64) * 2
        for codes, scales, scale in [(gu_codes, gu_scales, 1),
                                     (H // 4, gu_scales + H // 64 * 2, 2),
                                     (d_codes, d_scales, 0.5)]:
            file.seek(base + codes)
            file.write(bytes([0xE4]) * 16)
            file.seek(base + scales)
            file.write(struct.pack('<e', scale))
    for name, ids in [('valid', [9, 1, 4]), ('duplicate', [1, 1]), ('negative', [-1]), ('range', [16]), ('oversize', list(range(17)))]:
        (root / f'subset_{name}.bin').write_bytes(struct.pack('<' + 'i' * len(ids), *ids))
    (root / 'subset_empty.bin').write_bytes(b'')
    (root / 'subset_trailing.bin').write_bytes(b'\1\0\0')
    return lines, records


def main(test, info):
    with tempfile.TemporaryDirectory(prefix='strata-hpu-mtp-weights-') as directory:
        root = pathlib.Path(directory)
        lines, records = fixture(root)
        subprocess.run([test, str(root)], check=True)
        result = subprocess.run([info, str(root)], check=True, capture_output=True, text=True)
        assert 'no_device_acquired' in result.stdout
        gate_column = [line.replace('bf16 1 2560', 'bf16 2560 1') if line.startswith('mlp.shared_expert_gate.weight ') else line for line in lines]
        (root / 'dense.txt').write_text('\n'.join(gate_column) + '\n')
        subprocess.run([info, str(root)], check=True, capture_output=True, text=True)
        mutations = [(lines + [lines[0]], 'duplicate'), (lines[:-1], 'missing'),
                     ([lines[0].replace('q8_0', 'bad')] + lines[1:], 'unsupported'),
                     ([lines[0].replace('2560 2560', '2560 2559')] + lines[1:], 'geometry'),
                     ([lines[0] + ' extra'] + lines[1:], 'malformed'),
                     (['alias bf16 1 32 0 64'] + lines, 'overlapping')]
        for changed, expected in mutations:
            (root / 'dense.txt').write_text('\n'.join(changed) + '\n')
            failure = subprocess.run([info, str(root)], capture_output=True, text=True)
            assert failure.returncode and expected in failure.stderr, (changed[0], failure.stderr)
        (root / 'dense.txt').write_text('\n'.join(lines) + '\n')
        norm = next(record for record in records if record[0] == 'pre_fc_norm_embedding.weight')
        with (root / 'dense.bin').open('r+b') as file:
            file.seek(norm[4])
            file.write(struct.pack('<f', float('nan')))
        failure = subprocess.run([info, str(root)], capture_output=True, text=True)
        assert failure.returncode and 'non-finite MTP norm' in failure.stderr
        with (root / 'dense.bin').open('r+b') as file:
            file.seek(norm[4])
            file.write(struct.pack('<f', 1.25))
        with (root / 'experts.bin').open('r+b') as file:
            file.truncate(NE * BLOB - 1)
        failure = subprocess.run([info, str(root)], capture_output=True, text=True)
        assert failure.returncode and 'experts.bin size' in failure.stderr
    print('PASS no-card original MTP runtime preflight, duplicate/missing/overlap/kind/shape/trailing fields, finite norms and truncated expert guards')


if __name__ == '__main__':
    main(*sys.argv[1:])
