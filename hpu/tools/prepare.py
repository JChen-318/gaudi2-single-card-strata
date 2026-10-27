"""Create an opt-in HPU run config for the unchanged Strata API/web server.

Performs full native schema/memory preflight and exports the original complete
GGUF tokenizer/template. It neither acquires the card nor starts a server.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]


def first_shard(path):
    match = re.fullmatch(r'(.+)-\d{5}-of-(\d{5})\.gguf', path.name, re.IGNORECASE)
    return path.with_name(f'{match[1]}-00001-of-{match[2]}{path.suffix}') if match else path


def prepare(model, build, output, context=32768, module_id=None, *, mtp=None,
            mtp_draft_vocab=None, ple_gguf=None, expert_pack=None, control_vectors=(), engine_args=()):
    model, build, output = map(lambda path: pathlib.Path(path).resolve(), (model, build, output))
    if output.exists():
        raise ValueError(f'output already exists: {output}; choose a new directory')
    if not (build / 'strata-hpu').is_file():
        raise ValueError(f'native executable missing: {build / "strata-hpu"}')
    # Run exactly the eventual engine arguments through its own no-card mode.
    # The separate trunk-only plan tool cannot validate MTP, CPU dispatch or
    # generation/cache options. Reserve identity/mode switches to this helper.
    reserved = {'--native', '--max-context', '--module', '--serve', '--tokens',
                '--max-new', '--plan', '--plan-json', '--help', '--mtp', '--mtp-draft-vocab',
                '--control-vector', '--control-vector-scaled', '--ple-gguf', '--expert-pack'}
    engine_args = list(map(str, engine_args))
    if any(argument.split('=', 1)[0] in reserved for argument in engine_args):
        raise ValueError('engine_args cannot override model, execution mode, context, module or MTP paths')
    args = ['--native', str(model), '--max-context', str(context)]
    if module_id is not None:
        args += ['--module', str(module_id)]
    if ple_gguf is not None:
        args += ['--ple-gguf', str(pathlib.Path(ple_gguf).resolve())]
    if expert_pack is not None:
        args += ['--expert-pack', str(pathlib.Path(expert_pack).resolve())]
    if mtp is not None:
        args += ['--mtp', str(pathlib.Path(mtp).resolve())]
    if mtp_draft_vocab is not None:
        if mtp is None:
            raise ValueError('mtp_draft_vocab requires mtp')
        args += ['--mtp-draft-vocab', str(pathlib.Path(mtp_draft_vocab).resolve())]
    for path, scale in control_vectors:
        path = str(pathlib.Path(path).resolve())
        if ',' in path:
            raise ValueError('control vector paths cannot contain the native comma list separator')
        args += ['--control-vector-scaled', f'{path}:{scale}']
    args += engine_args
    plan = subprocess.run([str(build / 'strata-hpu'), *args, '--plan-json'],
                          check=True, capture_output=True, text=True)
    memory = json.loads(plan.stdout)
    if not memory.get('no_device_acquired'):
        raise ValueError('preflight did not report no-device mode')
    sys.path.insert(0, str(ROOT / 'tools'))
    from strata_tokenizer import extract
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='strata-hpu-prepare-', dir=output.parent) as temporary:
        staged = pathlib.Path(temporary) / 'run'
        staged.mkdir()
        tokenizer = extract(first_shard(model), staged)
        if not (staged / 'tokenizer' / 'chat_template.jinja').is_file():
            raise ValueError('GGUF lacks its chat template; refusing a generic replacement')
        config = {'exe': str(build / 'strata-hpu'), 'args': args, 'cwd': str(ROOT),
                  'tokenizer': str(output / 'tokenizer'), 'log': str(output / 'engine.log'),
                  'host': '127.0.0.1', 'lazy_load': True, 'open_browser': False,
                  'sampling': {'temperature': 0}, 'hpu_memory_plan': memory}
        (staged / 'strata-hpu.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        (staged / 'tokenizer-export.json').write_text(json.dumps(tokenizer, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        staged.rename(output)
    return output / 'strata-hpu.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=pathlib.Path)
    parser.add_argument('--build', required=True, type=pathlib.Path)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    parser.add_argument('--context', type=int, default=32768)
    parser.add_argument('--module', type=int)
    parser.add_argument('--mtp', type=pathlib.Path)
    parser.add_argument('--mtp-draft-vocab', type=pathlib.Path)
    parser.add_argument('--ple-gguf', type=pathlib.Path)
    parser.add_argument('--expert-pack', type=pathlib.Path)
    parser.add_argument('--control-vector', type=pathlib.Path, action='append', default=[])
    parser.add_argument('--control-vector-scaled', action='append', default=[])
    # --expert-profile decides WHICH experts --resident-cpu-experts/--resident-experts admit before routing,
    # so a resident run must be prepared with the same profile the engine will read.
    parser.add_argument('--expert-profile', type=pathlib.Path)
    parser.add_argument('--expert-profile-save', type=pathlib.Path)
    parser.add_argument('--control-vector-layer-range', type=int, nargs=2)
    # Native parsing performs range, combination, schema and memory checks.
    value_options = ['mtp-hnorm', 'mtp-max-t', 'mtp-expert-cache-mib', 'mtp-window', 'mtp-q4',
                     'spec', 'spec-min-p', 'suffix-draft', 'lookup-chain', 'lookup-chain-min', 'verify-window',
                     'prompt-cache', 'prompt-cache-every', 'snapshot-budget-mib', 'expert-cache-mib',
                     'prefill', 'short-read',
                     'hbm-limit-mib', 'pinned-limit-mib', 'host-limit-mib',
                     'cpu-cache-mib', 'cpu-workers', 'pool-workers', 'pool-tasks',
                     'hpu-load-budget', 'vram-reserve-mib',
                     'conversation-cache-min-free-mib', 'memory-limit-mib', 'cvec-mode', 'cvec-dir',
                     'prompt-cache-root', 'turn-token', 'tail-role-token', 'session-min-free-mib',
                     'resident-budget-gib', 'expert-profile-save-every', 'adapt-every', 'adapt-swaps',
                     'adapt-decay', 'adapt-min-gain', 'adapt-async']
    flag_options = ['spec-prob', 'coupled-draft', 'vram-elastic', 'cache-message-boundary', 'prompt-cache-tail', 'prefill-reference',
                    'mmap-experts', 'resident-experts', 'resident-cpu-experts', 'expert-cache-per-layer']
    for option in value_options:
        parser.add_argument('--' + option)
    for option in flag_options:
        parser.add_argument('--' + option, action='store_true')
    args = parser.parse_args()
    if args.context <= 0 or (args.module is not None and args.module < 0):
        parser.error('context must be positive and module must be nonnegative')
    try:
        native_args = []
        for option in value_options:
            value = getattr(args, option.replace('-', '_'))
            if value is not None:
                native_args += ['--' + option, value]
        for option in flag_options:
            if getattr(args, option.replace('-', '_')):
                native_args.append('--' + option)
        for option in ('expert_profile', 'expert_profile_save'):
            path = getattr(args, option)
            if path is not None:
                native_args += ['--' + option.replace('_', '-'), str(path.resolve())]
        if args.control_vector_layer_range is not None:
            native_args += ['--control-vector-layer-range', *map(str, args.control_vector_layer_range)]
        control_vectors = [(path, '1') for path in args.control_vector]
        for group in args.control_vector_scaled:
            for entry in group.split(','):
                path, separator, scale = entry.rpartition(':')
                if not separator or not path or not scale:
                    raise ValueError('--control-vector-scaled expects FILE:SCALE[,...]')
                control_vectors.append((path, scale))
        config = prepare(args.model, args.build, args.output, args.context, args.module,
                         mtp=args.mtp, mtp_draft_vocab=args.mtp_draft_vocab, ple_gguf=args.ple_gguf, expert_pack=args.expert_pack,
                         control_vectors=control_vectors, engine_args=native_args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        detail = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) and error.stderr else str(error)
        parser.exit(1, f'{detail}\n')
    print(f'Created {config}; no device acquired and no server started.')
    print(f'Server entry point: {sys.executable} -m serve.server --engine strata --config {config}')


if __name__ == '__main__':
    main()
