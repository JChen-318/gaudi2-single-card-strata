"""Full-vocabulary export and real no-card native preflight, without weights."""
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[2]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def main(build):
    fixtures = module(pathlib.Path(__file__).with_name('model_plan.py'), 'hpu_plan_fixture')
    prepare = module(ROOT / 'hpu/tools/prepare.py', 'hpu_prepare')
    with tempfile.TemporaryDirectory(prefix='strata-hpu-prepare-test-') as directory:
        base = pathlib.Path(directory)
        model = base / 'model.gguf'
        fixtures.fixture(model, tokenizer=True)
        config_path = prepare.prepare(model, build, base / 'run', context=128, module_id=7)
        config = json.loads(config_path.read_text())
        assert config['host'] == '127.0.0.1' and config['lazy_load'] and '--module' in config['args']
        tokenizer = pathlib.Path(config['tokenizer'])
        vocab = json.loads((tokenizer / 'vocab.json').read_text())
        assert len(vocab) == 248320 and vocab['token_248319'] == 248319
        sys.path.insert(0, str(ROOT / 'tools'))
        sys.path.insert(0, str(ROOT))
        from strata_tokenizer import Tokenizer
        from serve.frontend import ChatTemplate
        tokens = [None] * len(vocab)
        for token, number in vocab.items():
            tokens[number] = token
        engine_tokenizer = Tokenizer(tokens, [], json.loads((tokenizer / 'token_type.json').read_text()))
        text = 'Gaudi 中文🙂\n\t\x00'
        assert engine_tokenizer.decode(engine_tokenizer.encode(text)) == text
        template = ChatTemplate(tokenizer / 'chat_template.jinja')
        assert template.render([{'role': 'user', 'content': text}]) == text
        try:
            prepare.prepare(model, build, base / 'run')
        except ValueError:
            pass
        else:
            raise AssertionError('existing run directory overwritten')
        assert config['hpu_memory_plan']['no_device_acquired']
        # Exercise the public preparation CLI with real MTP files and an
        # unordered draft head. The emitted args must reproduce the exact plan.
        mtp_fixture = module(pathlib.Path(__file__).with_name('mtp_weights.py'), 'hpu_mtp_fixture')
        mtp = base / 'mtp runtime'
        mtp.mkdir()
        mtp_fixture.fixture(mtp)
        subset = mtp / 'subset_valid.bin'
        control_fixture = module(pathlib.Path(__file__).with_name('control_vector.py'), 'hpu_control_fixture')
        vector = base / 'control vector.gguf'
        control_fixture.fixture(vector, hidden=2560)
        run = base / 'mtp run'
        command = [sys.executable, str(ROOT / 'hpu/tools/prepare.py'), '--model', str(model),
                   '--build', str(build), '--output', str(run), '--context', '128', '--module', '7',
                   '--mtp', str(mtp), '--mtp-draft-vocab', str(subset), '--mtp-hnorm', 'stream', '--mtp-q4', 'all',
                   '--mtp-window', '64', '--mtp-max-t', '4', '--spec', '4', '--spec-min-p', '.25',
                   '--lookup-chain', '2', '--spec-prob', '--coupled-draft', '--prompt-cache', '2', '--verify-window', '3',
                   '--prompt-cache-every', '3', '--conversation-cache-min-free-mib', '2560',
                   '--prefill', '8', '--short-read', '2', '--prompt-cache-tail', '--prefill-reference',
                   '--memory-limit-mib', '32768',
                   '--session-min-free-mib', '4096',
                   '--prompt-cache-root', '8192', '--turn-token', '248045',
                   '--tail-role-token', '248046', '--cache-message-boundary',
                   '--control-vector-scaled', f'{vector}:.25', '--cvec-mode', 'project',
                   '--cvec-dir', 'single:1', '--control-vector-layer-range', '4', '44',
                   '--expert-cache-mib', '16', '--mtp-expert-cache-mib', '16']
        subprocess.run(command, check=True, capture_output=True, text=True)
        configured = json.loads((run / 'strata-hpu.json').read_text())
        assert configured['args'][configured['args'].index('--mtp') + 1] == str(mtp.resolve())
        assert configured['args'][configured['args'].index('--mtp-draft-vocab') + 1] == str(subset.resolve())
        assert '--spec-prob' in configured['args'] and '--coupled-draft' in configured['args']
        assert configured['args'][configured['args'].index('--prefill')+1]=='8'
        assert configured['args'][configured['args'].index('--short-read')+1]=='2' and '--prompt-cache-tail' in configured['args']
        assert '--prefill-reference' in configured['args']
        assert configured['args'][configured['args'].index('--verify-window')+1]=='3'
        assert configured['hpu_memory_plan']['verify_rows']==3 and configured['hpu_memory_plan']['window_rows']==3
        assert configured['args'][configured['args'].index('--mtp-q4') + 1] == 'all'
        assert configured['hpu_memory_plan']['mtp_q4_projections'] and configured['hpu_memory_plan']['mtp_q4_head']
        assert configured['args'][configured['args'].index('--memory-limit-mib') + 1] == '32768'
        assert configured['args'][configured['args'].index('--session-min-free-mib') + 1] == '4096'
        assert configured['args'][configured['args'].index('--prompt-cache-every') + 1] == '3'
        assert configured['args'][configured['args'].index('--prompt-cache-root') + 1] == '8192'
        assert '--cache-message-boundary' in configured['args']
        assert configured['args'][configured['args'].index('--control-vector-scaled') + 1] == f'{vector.resolve()}:.25'
        assert '--plan-json' not in configured['args'] and '--serve' not in configured['args']
        planned = subprocess.run([configured['exe'], *configured['args'], '--plan-json'],
                                 check=True, capture_output=True, text=True, cwd=configured['cwd'])
        assert json.loads(planned.stdout) == configured['hpu_memory_plan']
        assert configured['hpu_memory_plan']['mtp_state_bytes'] > 128 * 2 * 256 * 4
        assert configured['hpu_memory_plan']['mtp_cache_bytes'] == 16 << 20
        assert configured['hpu_memory_plan']['control_vector_host_bytes'] > 48 * 2560 * 4
        assert configured['hpu_memory_plan']['control_vector_hbm_bytes'] > 2560 * 4
        auto_run=base/'auto run'
        subprocess.run(command+['--prefill','auto','--output',str(auto_run)],check=True,capture_output=True,text=True)
        auto_config=json.loads((auto_run/'strata-hpu.json').read_text())
        assert auto_config['args'][auto_config['args'].index('--prefill')+1]=='auto'
        # The preceding command carries --prefill-reference. Auto still runs
        # the full optional-feature admission, then uses its row boundaries.
        assert auto_config['hpu_memory_plan']['prefill_rows']==0 and auto_config['hpu_memory_plan']['verify_rows']==3
        native_auto_args=[arg for arg in auto_config['args'] if arg!='--prefill-reference']
        auto_plan=json.loads(subprocess.run([auto_config['exe'],*native_auto_args,'--plan-json'],check=True,capture_output=True,text=True).stdout)
        assert auto_plan['prefill_rows']==128 and auto_plan['mtp_state_bytes']
        for resource in ['host','hbm']:
            before=auto_config['hpu_memory_plan']
            assert auto_plan[resource+'_required']-before[resource+'_required']==auto_plan['prefill_'+resource+'_bytes']-before['prefill_'+resource+'_bytes']+auto_plan['mtp_window_'+resource+'_bytes']-before['mtp_window_'+resource+'_bytes']
        external = base / 'external PLE-00002-of-00002.gguf'
        fixtures.external_ple_fixture(external)
        without_table = base / 'main-without-table.gguf'
        fixtures.fixture(without_table, missing='per_layer_token_embd.weight', tokenizer=True)
        external_run = base / 'external run'
        external_command = [sys.executable, str(ROOT / 'hpu/tools/prepare.py'), '--model', str(without_table),
                            '--build', str(build), '--output', str(external_run), '--context', '128', '--ple-gguf', str(external)]
        subprocess.run(external_command, check=True, capture_output=True, text=True)
        external_config = json.loads((external_run / 'strata-hpu.json').read_text())
        assert external_config['args'][external_config['args'].index('--ple-gguf')+1] == str(external.resolve())
        assert external_config['hpu_memory_plan']['no_device_acquired']
        from native_pack_model import fixture as pack_fixture
        pack=base/'native expert pack'
        pack_fixture(pack,without_table)
        pack_run=base/'pack run'
        pack_command=[*external_command,'--expert-pack',str(pack),'--output',str(pack_run)]
        subprocess.run(pack_command,check=True,capture_output=True,text=True)
        pack_config=json.loads((pack_run/'strata-hpu.json').read_text())
        assert pack_config['args'][pack_config['args'].index('--expert-pack')+1]==str(pack.resolve())
        assert pack_config['hpu_memory_plan']['native_expert_pack_host_bytes']==4<<20
        # Rejected configuration never starts an engine or publishes an output
        # directory. Mode/identity switches cannot bypass no-card preparation.
        for kwargs in [dict(engine_args=['--serve']), dict(engine_args=['--native=other.gguf']),
                       dict(engine_args=['--ple-gguf', str(external)]), dict(ple_gguf=base / 'missing-ple.gguf'),
                       dict(engine_args=['--expert-pack',str(pack)]),dict(expert_pack=base/'missing-pack'),
                       dict(mtp_draft_vocab=subset), dict(mtp=base / 'missing runtime'),
                       dict(engine_args=['--mtp-q4', 'all']), dict(mtp=mtp, engine_args=['--mtp-q4', 'invalid']),
                       dict(engine_args=['--hbm-limit-mib', '1']),
                       dict(engine_args=['--prefill','32769']),dict(engine_args=['--verify-window','8']),dict(engine_args=['--short-read','2147483648']),
                       dict(engine_args=['--hpu-load-budget', '0', '--pool-workers', '-1']),
                       dict(engine_args=['--hpu-load-budget', '0', '--pool-tasks', '4097']),
                       dict(engine_args=['--hpu-load-budget', '0', '--cpu-cache-mib', '0']),
                       dict(engine_args=['--turn-token', '-2']),
                       dict(engine_args=['--tail-role-token', '-2']),
                       dict(engine_args=['--prompt-cache-root', '4294967295']),
                       dict(engine_args=['--unknown-hpu-setting', '1'])]:
            rejected = base / 'rejected'
            try:
                prepare.prepare(model, build, rejected, context=128, **kwargs)
            except (ValueError, subprocess.CalledProcessError):
                pass
            else:
                raise AssertionError(f'invalid preparation accepted: {kwargs}')
            assert not rejected.exists()
        failure = subprocess.run(command + ['--hbm-limit-mib', '1', '--output', str(base / 'bad-cli')],
                                 capture_output=True, text=True)
        assert failure.returncode and 'memory policy' in failure.stderr
        assert not (base / 'bad-cli').exists()
    print('PASS exact no-card engine configuration including MTP/subset/cache and budget guards; all 248320 vocabulary entries, Unicode/template and atomic config creation')


if __name__ == '__main__':
    main(pathlib.Path(sys.argv[1]).resolve())
