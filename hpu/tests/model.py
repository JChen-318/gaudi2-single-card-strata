"""Hardware-only complete native model acceptance, sparse zero block weights."""
import pathlib
import subprocess
import struct
import sys
import tempfile
import os
from model_plan import fixture, external_ple_fixture

with tempfile.TemporaryDirectory(prefix='strata-hpu-native-model-') as directory:
    root = pathlib.Path(directory)
    model = root / 'model.gguf'
    external_ple = '--external-ple' in sys.argv[2:]
    locations = fixture(model, head_type=2 if '--packed-head' in sys.argv[2:] else 30,
                        missing='per_layer_token_embd.weight' if external_ple else None)
    environment = os.environ.copy()
    if external_ple:
        external = root / 'external-00002-of-00002.gguf'
        external_ple_fixture(external)
        environment['STRATA_HPU_TEST_PLE_GGUF'] = str(external)
    # Every zero-weight block preserves its input residual. Distinct, exactly
    # representable embedding rows make stale/partially published residuals
    # observable while the independent output-head oracle remains all zero.
    with model.open('r+b') as file:
        for token in range(11):
            file.seek(locations['token_embd.weight'] + token * 2560 * 2)
            row = [struct.unpack('<I', struct.pack('<f', token * (i % 7 - 3) / 8))[0] >> 16 for i in range(2560)]
            file.write(struct.pack('<2560H', *row))
    command = [sys.argv[1], str(model), str(root / 'session.bin')]
    if '--control' in sys.argv[2:]:
        sys.dont_write_bytecode = True
        from control_vector import fixture as control_fixture
        control = root / 'control.gguf'
        control_fixture(control, hidden=2560, rows=[('direction.47', [1.] + [0.] * 2559)])
        command.append(str(control))
    if '--mtp' in sys.argv[2:] or '--mtp-subset' in sys.argv[2:]:
        sys.dont_write_bytecode = True
        from mtp_weights import fixture as mtp_fixture
        runtime = root / 'mtp'
        runtime.mkdir()
        _, records = mtp_fixture(runtime)
        if '--mtp-subset' in sys.argv[2:]:
            (runtime / 'draft_vocab.bin').write_bytes(struct.pack('<3i', 10, 0, 2))
            # Independent nonzero subset-head oracle: zero block weights leave
            # a single positive FC-embedding feature. Final HC mix contributes
            # 0.5*sqrt(2560), rounded to BF16 25.25; head rows below yield
            # logits {10:25.25, 0:50.5, 2:-25.25}. Target final norm stays zero.
            norm = next(row for row in records if row[0] == 'hyper_connection_mixer.hc_norm.weight')
            with (runtime / 'dense.bin').open('r+b') as file:
                file.seek(norm[4])
                file.write(struct.pack('<10240f', *([1.0] * 10240)))
            with model.open('r+b') as file:
                for token, weight in [(10, 1.0), (0, 2.0), (2, -1.0)]:
                    file.seek(locations['output.weight'] + token * 2560 * 2)
                    file.write(struct.pack('<H', struct.unpack('<I', struct.pack('<f', weight))[0] >> 16))
        command.append(str(runtime))
        for mode in ['all', 'proj', 'head']:
            if '--mtp-q4-' + mode in sys.argv[2:]:
                command.append(mode)
    if '--expert-pack-gguf' in sys.argv[2:] or '--expert-pack-bin' in sys.argv[2:]:
        from native_pack_model import fixture as pack_fixture
        pack = root / 'native-expert-pack'
        pack_fixture(pack,model,binary='--expert-pack-bin' in sys.argv[2:])
        environment['STRATA_HPU_TEST_EXPERT_PACK']=str(pack)
    subprocess.run(command, check=True, env=environment)
