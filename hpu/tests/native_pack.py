"""Actual native source with independently packed bytes, provenance and bad inputs."""
import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import os
from types import SimpleNamespace


def fixture(root, *, mixed=True, version=4, binary=True, nv=True):
    root.mkdir(exist_ok=True)
    names = ['model-00001-of-00002.gguf', 'model-00002-of-00002.gguf']
    files = [bytearray(128), bytearray(128)]
    lines, blobs, offset = [], [], 0
    for layer in range(3):
        role_offsets, role_files, matrices = [], [], []
        for role in range(3):
            shard = int(layer == 1 or (layer == 0 and mixed and role == 1) or (layer == 2 and mixed and role == 2))
            kind = 42 if layer == 0 else 8 if layer == 1 or role == 2 or not nv else 40
            size = 64 * (18 if kind == 42 else 36 if kind == 40 else 68)
            role_offsets.append(len(files[shard]))
            role_files.append('' if shard == 0 else names[shard])
            for expert in range(3):
                files[shard] += bytes([7 + layer * 9 + role * 3 + expert]) * size
            matrices.append(size)
        blob_size = sum(matrices) + (16 if layer == 2 and nv else 0)
        row = f'{layer} {42 if layer == 0 else 8 if layer == 1 or not nv else 40} {42 if layer == 0 else 8} {offset} {blob_size}'
        if (layer != 2 or not nv) and version >= 2:
            row += ' ' + ' '.join(map(str, role_offsets))
            if len(set(role_files)) == 1:
                if role_files[0]:
                    row += ' ' + role_files[0]
            else:
                row += ' ' + ','.join(role_files)
        lines.append(row)
        for expert in range(3):
            for role, size in enumerate(matrices):
                blobs.append(bytes([31 + layer * 21 + role * 5 + expert]) * size)
            if layer == 2 and nv:
                blobs.append(struct.pack('<4f', .25, .5, 1, 0))
        offset += blob_size * 3
    text = f'# strata native experts v{version}: layer gu_type d_type offset blob_bytes (n_expert 3, total {offset}; absolute offsets)\n' + '\n'.join(lines) + '\n'
    (root / 'native_experts.txt').write_text(text, encoding='utf-8', newline='\n')
    for name, data in zip(names, files):
        (root / name).write_bytes(data)
    if binary:
        (root / 'experts.bin').write_bytes(b''.join(blobs))
        provenance = {'schema': 1, 'shards': [{'name': name, 'size': len(data)} for name, data in zip(names, files)],
                      'native_experts_sha256': hashlib.sha256(text.encode()).hexdigest(), 'bytes': offset}
        (root / 'experts.bin.src.json').write_text(json.dumps(provenance) + '\n')
    return text


def original_producer(root, *, mixed=True, primary=0):
    # Execute the unchanged producer against the same tiny explicit catalog.
    # This exercises iq_pack.py's v3/v4 columns and original provenance writer;
    # actual full-GGUF Model parsing is covered by the Linux integration helper.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'tools'))
    import iq_pack
    paths = [root / 'model-00001-of-00002.gguf', root / 'model-00002-of-00002.gguf']
    model = SimpleNamespace(paths=paths, sizes=[p.stat().st_size for p in paths], where={})
    position = [128, 128]
    for layer in range(3):
        model.where[f'blk.{layer}.ffn_gate_inp.weight'] = (None, SimpleNamespace(shape=[64, 3]), None, paths[0])
        for role, name in enumerate(['gate', 'up', 'down']):
            shard = int(layer == 1 or (layer == 0 and mixed and role == 1) or (layer == 2 and mixed and role == 2))
            kind = 42 if layer == 0 else 8 if layer == 1 or role == 2 else 40
            size = 64 * (18 if kind == 42 else 36 if kind == 40 else 68)
            tensor = SimpleNamespace(shape=[64,64,3], type_id=kind, type_name={42:'Q2_0',8:'Q8_0',40:'NVFP4'}[kind],
                                     offset=position[shard]-128, expected_bytes=lambda size=size: size*3)
            model.where[f'blk.{layer}.ffn_{name}_exps.weight'] = (SimpleNamespace(data_start=128), tensor, None, paths[shard])
            if kind == 40:
                model.where[f'blk.{layer}.ffn_{name}_exps.scale'] = (None, SimpleNamespace(shape=[3]), None, paths[shard])
            position[shard] += size * 3
    result = iq_pack.expert_layout(model, paths[primary])
    assert not isinstance(result, str), result
    _, text, count, total = result
    assert text.startswith('# strata native experts v4:' if mixed else '# strata native experts v3:')
    assert count == 3 and total == (root / 'experts.bin').stat().st_size
    (root / 'native_experts.txt').write_text(text, encoding='utf-8', newline='\n')
    (root / 'experts.bin.src.json').write_text(json.dumps(iq_pack.experts_source(model, text, total)))


def main(executable):
    def run(root, mixed=True, success=True, nv=True):
        result = subprocess.run([executable, str(root), ('v4' if mixed else 'common') + ('' if nv else '-flat')], capture_output=True, text=True)
        assert (result.returncode == 0) == success, result.stdout + result.stderr
        return result

    with tempfile.TemporaryDirectory(prefix='strata-hpu-native-pack-') as directory:
        root = pathlib.Path(directory)
        for version in [1, 2, 3, 4]:
            pack = root / str(version)
            fixture(pack, version=version, mixed=version == 4)
            run(pack, mixed=version == 4)
        for mixed in [False, True]:
            producer = root / ('producer-v4' if mixed else 'producer-v3')
            fixture(producer, mixed=mixed)
            original_producer(producer, mixed=mixed)
            run(producer, mixed=mixed)
            original_producer(producer, mixed=mixed, primary=1)
            run(producer, mixed=mixed)
        for binary in [True, False]:
            fallback = root / ('flat-bin' if binary else 'flat-gguf')
            fixture(fallback, binary=binary, nv=False)
            run(fallback, nv=False)
        pack = root / 'current'
        text = fixture(pack)
        first = run(pack).stdout
        old = (pack / 'native_experts.txt').stat()
        replacement = pack / 'replacement'
        replacement.write_bytes((pack / 'native_experts.txt').read_bytes())
        os.utime(replacement, ns=(old.st_atime_ns, old.st_mtime_ns))
        os.replace(replacement, pack / 'native_experts.txt')
        assert run(pack).stdout != first, 'native pack identity omitted same-size/mtime file replacement'
        # No binary: every layer needs source offsets. NVFP4's five-field row
        # cannot supply its tails, so it must fail before returning any expert.
        (pack / 'experts.bin').unlink()
        run(pack, success=False)
        # Separate all-Q2/Q8 no-binary case is exercised in the GGUF integration
        # fixture, where main-model scale-free source offsets are available.
        def wrong_offset(s):
            lines=s.splitlines()
            row=lines[1].split()
            row[5]=str(int(row[5])+1)
            lines[1]=' '.join(row)
            return '\n'.join(lines)+'\n'
        for mutation in [wrong_offset, lambda s: s.replace('v4:', 'v5:'), lambda s: s.replace('(n_expert 3,', '(n_expert 4,'),
                         lambda s: s.replace('0 42 42 0 ', '0 42 42 1 '),
                         lambda s: s + s.splitlines()[1] + '\n', lambda s: '\n'.join(s.splitlines()[:-1]) + '\n',
                         lambda s: s.replace('model-00002-of-00002.gguf,', 'missing.gguf,'),
                         lambda s: s.replace('0 42 42 0 ', '0 8 42 0 '),
                         lambda s: s.replace('v4:', 'v3:')]:
            original = fixture(pack)
            changed = mutation(original)
            if changed == original:
                continue
            (pack / 'native_experts.txt').write_text(changed, encoding='utf-8')
            run(pack, success=False)
        for change in ['digest', 'shard', 'size', 'bytes', 'schema', 'missing', 'duplicate', 'unknown']:
            fixture(pack)
            path = pack / 'experts.bin.src.json'
            source = json.loads(path.read_text())
            if change == 'digest': source['native_experts_sha256'] = '0' * 64
            if change == 'shard': source['shards'][1]['name'] = 'other.gguf'
            if change == 'size': source['shards'][1]['size'] += 1
            if change == 'bytes': source['bytes'] += 1
            if change == 'schema': source['schema'] = 2
            if change == 'unknown': source['unexpected'] = 1
            if change == 'missing': path.unlink()
            else:
                raw = json.dumps(source)
                if change == 'duplicate': raw = raw[:-1] + ', "schema": 1}'
                path.write_text(raw)
            run(pack, success=False)
        for offset, value in [(-16, 1.), (-12, float('nan')), (-8, .5), (-4, -0.)]:
            fixture(pack)
            with (pack / 'experts.bin').open('r+b') as file:
                file.seek(offset, 2)
                file.write(struct.pack('<f', value))
            run(pack, success=False)
        fixture(pack)
        with (pack / 'experts.bin').open('r+b') as file: file.truncate(file.seek(0, 2) - 1)
        run(pack, success=False)
        # Padding boundaries and binary bytes checked against Python hashlib.
        raw = root / 'hash-input'
        for length in [0, 3, 55, 56, 63, 64, 65, 127, 128, 1024, 65536]:
            data = bytes(i % 251 for i in range(length))
            raw.write_bytes(data)
            actual = subprocess.run([executable, '--sha256', str(raw)], capture_output=True, text=True, check=True).stdout.strip()
            assert actual == hashlib.sha256(data).hexdigest()
        valid = {'schema': 1, 'bytes': 1, 'native_experts_sha256': '0' * 64, 'shards': [{'name': '中文🙂.gguf', 'size': 2**63+1}]}
        for ascii_only in [False, True]:
            raw.write_text(json.dumps(valid, ensure_ascii=ascii_only), encoding='utf-8')
            result = subprocess.run([executable, '--json', str(raw)], capture_output=True, encoding='utf-8', check=True)
            assert result.stdout.strip() == '中文🙂.gguf'
        for bad in ['{}', '{"schema":01}', json.dumps(valid).replace('9223372036854775809', '18446744073709551616'),
                    json.dumps(valid).replace('"schema": 1', '"schema": 1.0'),
                    json.dumps(valid).replace('\\ud83d\\ude42', '\\ud83d'), json.dumps(valid) + 'junk']:
            raw.write_text(bad, encoding='utf-8')
            assert subprocess.run([executable, '--json', str(raw)], capture_output=True).returncode
    print('PASS actual native pack v1-v4 binary selection, role offsets, tails, provenance, SHA256/Unicode/uint64 parsing, ownership and rejection corpus; no device')


if __name__ == '__main__':
    main(sys.argv[1])
