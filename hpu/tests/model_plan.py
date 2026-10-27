"""Sparse, complete 48-layer GGUF directory fixture; never reads model payloads.

This tests the real plan executable without SDK initialization. The logical
file is large; only its header and a final byte occupy disk blocks.
"""
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile


def mark_sparse(file):
    # Windows host fixture validation must not allocate tens of GiB of zeros.
    # Linux seek/truncate already creates holes. Never use this for user files.
    if sys.platform == 'win32':
        import ctypes
        import msvcrt
        from ctypes import wintypes
        call = ctypes.WinDLL('kernel32', use_last_error=True).DeviceIoControl
        call.argtypes = [wintypes.HANDLE,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,
                         wintypes.LPVOID,wintypes.DWORD,ctypes.POINTER(wintypes.DWORD),wintypes.LPVOID]
        call.restype = wintypes.BOOL
        returned = wintypes.DWORD()
        if not call(msvcrt.get_osfhandle(file.fileno()),0x900c4,None,0,None,0,ctypes.byref(returned),None):
            raise ctypes.WinError(ctypes.get_last_error())


def allocated_bytes(path):
    if sys.platform != 'win32':
        return path.stat().st_blocks*512
    import ctypes
    from ctypes import wintypes
    call=ctypes.WinDLL('kernel32',use_last_error=True).GetCompressedFileSizeW
    call.argtypes=[wintypes.LPCWSTR,ctypes.POINTER(wintypes.DWORD)]
    call.restype=wintypes.DWORD
    upper=wintypes.DWORD()
    ctypes.set_last_error(0)
    lower=call(str(path),ctypes.byref(upper))
    if lower==0xffffffff and ctypes.get_last_error():
        raise ctypes.WinError(ctypes.get_last_error())
    return (upper.value<<32)|lower


def text(value):
    encoded = value.encode()
    return struct.pack('<Q', len(encoded)) + encoded


def fixture(path, missing=None, bad_shape=False, tokenizer=False, head_type=30, nvfp4_roles=(), expert_type=42):
    mult = [23703573157769, 20109073645365, 8052911324071]
    vocab = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063,
             20000069, 20000077, 20000081, 20000093, 20000107, 20000147,
             20000153, 20000159, 20000161, 20000171]
    offsets, running = [], 0
    for size in vocab:
        offsets.append(running)
        running += size
    metadata = [('general.architecture', 8, text('qwen4exp'))]
    metadata.append(('tokenizer.ggml.eos_token_id', 4, struct.pack('<I', 248044)))
    integers = {'block_count': 48, 'embedding_length': 2560, 'expert_count': 512,
                'expert_used_count': 10, 'attention.head_count': 24, 'attention.head_count_kv': 2,
                'ple.ngram_size': 3, 'ple.heads_per_ngram': 8, 'ple.conv_kernel': 4,
                'ple.eos_token_id': 248044, 'embedding_length_per_layer_input': 160}
    metadata += [('qwen4exp.' + key, 4, struct.pack('<I', value)) for key, value in integers.items()]
    for key, values in [('ple.layers', [1]), ('ple.layer_multipliers', mult),
                        ('ple.head_vocab_sizes', vocab), ('ple.head_offsets', offsets)]:
        metadata.append(('qwen4exp.' + key, 9, struct.pack('<IQ', 10, len(values)) + struct.pack('<' + 'Q' * len(values), *values)))
    if tokenizer:
        root = pathlib.Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / 'tools'))
        from strata_tokenizer import BYTE_TO_UNICODE
        tokens = [BYTE_TO_UNICODE[b] for b in range(256)] + [f'token_{i}' for i in range(256, 248320)]
        metadata += [('tokenizer.ggml.model', 8, text('gpt2')),
                     ('tokenizer.ggml.pre', 8, text('qwen35')),
                     ('tokenizer.ggml.tokens', 9, struct.pack('<IQ', 8, len(tokens)) + b''.join(text(token) for token in tokens)),
                     ('tokenizer.ggml.merges', 9, struct.pack('<IQ', 8, 0)),
                     ('tokenizer.ggml.token_type', 9, struct.pack('<IQ', 5, len(tokens)) + struct.pack('<i', 1) * len(tokens)),
                     ('tokenizer.chat_template', 8, text('{% for message in messages %}{{ message.content }}{% endfor %}'))]
    tensors = []

    def tensor(name, shape, kind=30):
        if name == missing:
            return
        if bad_shape and name == 'blk.47.attn_output.weight':
            shape = [6144, 2559]
        tensors.append((name, shape, kind))

    tensor('token_embd.weight', [2560, 248320])
    tensor('output.weight', [2560, 248320], head_type)
    tensor('output_hc_norm.weight', [2560, 4], 0)
    tensor('output_hc_down.weight', [10240, 320])
    tensor('output_hc_up.weight', [320, 10240])
    for layer in range(48):
        prefix = f'blk.{layer}.'
        for half in ['hc_attn_', 'hc_ffn_']:
            for suffix, shape, kind in [('down.weight', [10240, 320], 30), ('up.weight', [320, 10240], 30),
                                        ('inject.weight', [10240, 4], 30), ('norm.weight', [2560, 4], 0)]:
                tensor(prefix + half + suffix, shape, kind)
        tensor(prefix + 'ffn_gate_inp.weight', [2560, 512])
        tensor(prefix + 'ffn_gate_inp_shexp.weight', [2560])
        for suffix, shape in [('gate', [2560, 640]), ('up', [2560, 640]), ('down', [640, 2560])]:
            tensor(prefix + 'ffn_' + suffix + '_exps.weight', shape + [512], 40 if layer==0 and suffix in nvfp4_roles else expert_type)
            if layer==0 and suffix in nvfp4_roles:
                tensor(prefix + 'ffn_' + suffix + '_exps.scale', [512], 0)
            tensor(prefix + 'ffn_' + suffix + '_shexp.weight', shape, expert_type)
        if layer % 4 == 3:
            for suffix, shape in [('attn_q.weight', [2560, 12288]), ('attn_k.weight', [2560, 512]),
                                  ('attn_v.weight', [2560, 512]), ('attn_output.weight', [6144, 2560]),
                                  ('indexer.q_proj.weight', [2560, 512]), ('indexer.k_proj.weight', [2560, 128])]:
                tensor(prefix + suffix, shape)
            for suffix, width in [('attn_q_norm.weight', 256), ('attn_k_norm.weight', 256),
                                  ('indexer.q_norm.weight', 128), ('indexer.k_norm.weight', 128)]:
                tensor(prefix + suffix, [width], 0)
        else:
            for suffix, shape in [('attn_qkv.weight', [2560, 10240]), ('attn_gate.weight', [2560, 6144]),
                                  ('ssm_out.weight', [6144, 2560]), ('ssm_alpha.weight', [2560, 48]), ('ssm_beta.weight', [2560, 48])]:
                tensor(prefix + suffix, shape)
            for suffix, shape in [('ssm_norm.weight', [128]), ('ssm_dt.bias', [48]), ('ssm_a', [48]), ('ssm_conv1d.weight', [4, 10240])]:
                tensor(prefix + suffix, shape, 0)
        if layer == 1:
            tensor(prefix + 'ple_key.weight', [2560, 10240])
            tensor(prefix + 'ple_value.weight', [2560, 2560])
            for suffix in ['key', 'query', 'conv']:
                tensor(prefix + 'ple_norm_' + suffix + '.weight', [2560, 4], 0)
            tensor(prefix + 'ple_conv1d.weight', [4, 10240], 0)
    tensor('per_layer_token_embd.weight', [160, 320001536], 20)
    header = bytearray(struct.pack('<IIQQ', 0x46554747, 3, len(tensors), len(metadata)))
    for key, kind, payload in metadata:
        header += text(key) + struct.pack('<I', kind) + payload
    offset = 0
    locations = {}
    for name, shape, kind in tensors:
        locations[name] = offset
        header += text(name) + struct.pack('<I', len(shape)) + struct.pack('<' + 'Q' * len(shape), *shape) + struct.pack('<IQ', kind, offset)
        elements = 1
        for dimension in shape:
            elements *= dimension
        # GGML (values per block, block bytes). K-quants use 256-value superblocks,
        # so a head at embedding length 2560 keeps ten 144/176-byte blocks per row.
        blocks = {0: (1, 4), 30: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
                8: (32, 34), 20: (32, 18), 40: (64, 36), 42: (64, 18), 11: (256, 110),
                12: (256, 144), 13: (256, 176), 14: (256, 210), 16: (256, 66), 17: (256, 74), 18: (256, 98), 21: (256, 110), 22: (256, 82), 23: (256, 136), 29: (256, 56), 40: (64, 36)}
        if kind not in blocks:
            raise AssertionError('model_plan fixture lacks GGML type %d' % kind)
        block, block_bytes = blocks[kind]
        size = elements // block * block_bytes
        offset = (offset + size + 31) // 32 * 32
    header += bytes((-len(header)) % 32)
    with path.open('wb') as stream:
        mark_sparse(stream)
        stream.write(header)
        stream.seek(len(header) + offset - 1)
        stream.write(b'\0')
        for role in nvfp4_roles:
            stream.seek(len(header)+locations['blk.0.ffn_'+role+'_exps.scale'])
            stream.write(struct.pack('<512f', *[float((i%7)+1)/8 for i in range(512)]))
    return {name: len(header) + location for name, location in locations.items()}


def external_ple_fixture(path, rows=320001536):
    # Linux host/integration fixture: huge logical table, zero resident payload.
    # Only the directory is written; original bounded Direct I/O gathers zeros.
    header = struct.pack('<IIQQ', 0x46554747, 3, 1, 0)
    header += text('per_layer_token_embd.weight') + struct.pack('<IQQIQ', 2, 160, rows, 20, 0)
    header += bytes((-len(header)) % 32)
    with path.open('wb') as file:
        mark_sparse(file)
        file.write(header)
        if rows:
            file.seek(len(header) + rows * 90 - 1)
            file.write(b'\0')


def main(executable):
    def check_auto(command):
        # Call the actual native entry point with the complete optional feature
        # set. Limits rounded down to MiB must select the largest admissible
        # row count, never admit against the cheaper trunk-only baseline.
        def run(extra=()):
            return json.loads(subprocess.run(command+list(extra)+['--plan-json'],capture_output=True,text=True,check=True).stdout)
        baseline=run(['--prefill','0'])
        maximum=run(['--prefill','128'])
        assert run(['--prefill','auto'])==maximum
        assert run(['--prefill','auto','--prefill-reference'])==baseline
        assert run(['--prefill','auto','--prefill','8'])==run(['--prefill','8'])
        assert run(['--prefill','8','--prefill','auto'])==maximum
        for resource in ['host','hbm']:
            full=maximum[resource+'_required']
            # Test multiple limits including less than the graph fixed cost:
            # fallback may select0; every row above the choice must be refused.
            for numerator in [0,1,3,7]:
                desired=baseline[resource+'_required']+(full-baseline[resource+'_required'])*numerator//8
                limit=(desired+(1<<20)-1)//(1<<20)
                extra=[f'--{resource}-limit-mib',str(limit)]
                selected=run(['--prefill','auto']+extra)
                rows=selected['prefill_rows']
                assert selected[resource+'_required']<=limit*(1<<20)
                assert selected==run(['--prefill',str(rows)]+extra)
                if rows<128:
                    rejected=subprocess.run(command+['--plan-json','--prefill',str(rows+1)]+extra,capture_output=True,text=True)
                    assert rejected.returncode and 'memory policy' in rejected.stderr
                assert run(['--prefill','auto','--prefill-reference']+extra)==baseline
            too_small=(baseline[resource+'_required']-1)//(1<<20)
            rejected=subprocess.run(command+['--plan-json','--prefill','auto',f'--{resource}-limit-mib',str(too_small)],capture_output=True,text=True)
            assert rejected.returncode and 'memory policy' in rejected.stderr
    with tempfile.TemporaryDirectory(prefix='strata-hpu-model-plan-') as directory:
        path = pathlib.Path(directory) / 'model.gguf'
        fixture(path)
        result = subprocess.run([executable, str(path)], capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        assert plan['layers'] == 48 and plan['no_device_acquired']
        # Independent sums for BF16 layer payloads, F32 constants, and head.
        hc = 2 * (2 * 10240 * 320 * 2 + 10240 * 4 * 2 + 10240 * 4)
        router = 2560 * 513 * 2
        gdn = (2560 * (10240 + 6144 + 6144 + 96) * 2 + (128 + 96 + 4 * 10240) * 4)
        qsa = 2560 * (12288 + 512 + 512 + 6144 + 512 + 128) * 2
        ple = (2560 * 10240 + 2560 * 2560) * 2 + (3 * 10240 + 4 * 10240) * 4
        head = 2560 * 248320 * 2 + 10240 * 4 + 2 * 10240 * 320 * 2
        assert plan['head_bytes'] == head
        assert plan['packed_expert_execution_bytes'] == (((26 * 2560 + 37 * 640) // 4 * 3 + 4 * (2 * 2560 + 640)) * (1 + 8 + 32 + 128) + 25600 * 3 * 3 * 4) * 48
        assert plan['dense_bytes'] == 48 * (hc + router) + 36 * gdn + 12 * qsa + ple + head
        shorter = json.loads(subprocess.run([executable, str(path), '128'], capture_output=True, text=True, check=True).stdout)
        assert shorter['host_state_bytes'] < plan['host_state_bytes']
        external = pathlib.Path(directory) / 'external PLE-00002-of-00002.gguf'
        external_ple_fixture(external)
        without_table = pathlib.Path(directory) / 'main-without-table.gguf'
        fixture(without_table, missing='per_layer_token_embd.weight')
        external_plan = subprocess.run([executable, str(without_table), '128', str(external)], capture_output=True, text=True, check=True)
        assert json.loads(external_plan.stdout) == shorter
        assert subprocess.run([executable, str(without_table)], capture_output=True, text=True).returncode
        small = pathlib.Path(directory) / 'small-table.gguf'
        external_ple_fixture(small, rows=32)
        assert subprocess.run([executable, str(without_table), '128', str(small)], capture_output=True, text=True).returncode
        if len(sys.argv) > 2:
            external_command = [sys.argv[2], '--native', str(without_table), '--plan-json', '--max-context', '128', '--ple-gguf', str(external)]
            assert json.loads(subprocess.run(external_command, capture_output=True, text=True, check=True).stdout) == shorter
            for mode in ['direct','mmap']:
                expected=dict(shorter)
                if mode=='mmap':
                    expected.update(ple_reader_cache_rows=0,ple_reader_worker_bytes=0)
                assert json.loads(subprocess.run(external_command+['--ple-io',mode],capture_output=True,text=True,check=True).stdout)==expected
            large=json.loads(subprocess.run(external_command+['--ple-row-cache',str(8<<20),'--ple-inflight','1024',
                '--host-limit-mib','16384','--ple-delay-us','5000'],capture_output=True,text=True,check=True).stdout)
            assert large['ple_reader_cache_rows']==8<<20
            assert large['ple_reader_extra_host_bytes']>0
            assert large['host_required']==shorter['host_required']+large['ple_reader_host_bytes']-shorter['ple_reader_host_bytes']
            assert large['hbm_required']==shorter['hbm_required'] and large['no_device_acquired']
            huge=subprocess.run(external_command+['--ple-row-cache',str((1<<63)-1)],capture_output=True,text=True)
            assert huge.returncode and 'reservation overflow' in huge.stderr
            refused_ram=subprocess.run(external_command+['--ple-io','ram'],capture_output=True,text=True)
            assert refused_ram.returncode and 'full-table RAM reservation' in refused_ram.stderr
            ram_flags=['--ple-io','ram','--host-limit-mib','65536']
            ram=json.loads(subprocess.run(external_command+ram_flags,capture_output=True,text=True,check=True).stdout)
            assert ram['ple_ram_table_bytes']==320001536*90
            assert ram['ple_ram_host_bytes']>ram['ple_ram_table_bytes']
            assert ram['host_required']==shorter['host_required']+ram['ple_ram_host_bytes']+ram['ple_reader_host_bytes']-shorter['ple_reader_host_bytes']
            assert ram['hbm_required']==shorter['hbm_required'] and ram['no_device_acquired']
            assert ram['host_state_bytes']==shorter['host_state_bytes'] and ram['control_bytes']==shorter['control_bytes']
            below=(ram['host_required']-1)//(1<<20)
            rejected_ram=subprocess.run(external_command+['--ple-io','ram','--host-limit-mib',str(below)],capture_output=True,text=True)
            assert rejected_ram.returncode and 'full-table RAM reservation' in rejected_ram.stderr
            admitted=(ram['host_required']+(1<<20)-1)//(1<<20)
            assert json.loads(subprocess.run(external_command+['--ple-io','ram','--host-limit-mib',str(admitted)],capture_output=True,text=True,check=True).stdout)==ram
            # Diagnostic buffers must be enabled BEFORE CLI --plan returns;
            # the old late assignment omitted even the layer ladder's charge.
            route_file=pathlib.Path(directory)/'routing-plan-only.bin'
            layer_file=pathlib.Path(directory)/'layers-plan-only.bin'
            routed=json.loads(subprocess.run(external_command+['--dump-routing',str(route_file)],capture_output=True,text=True,check=True).stdout)
            charge=routed['routing_capture_host_bytes']
            assert charge>4096 and routed['dump_host_bytes']==charge
            assert routed['host_required']-shorter['host_required']==charge and routed['control_bytes']-shorter['control_bytes']==charge
            assert routed['hbm_required']==shorter['hbm_required'] and routed['no_device_acquired'] and not route_file.exists()
            resident_plan=json.loads(subprocess.run(external_command+['--serve','--dump-routing',str(route_file)],capture_output=True,text=True,check=True).stdout)
            assert resident_plan==routed and not route_file.exists()
            half_file=pathlib.Path(directory)/'halves-plan-only.bin'
            half_flags=['--dump-halves',str(half_file)]
            half_plan=json.loads(subprocess.run(external_command+half_flags,capture_output=True,text=True,check=True).stdout)
            # Independent upstream wire extent plus owned frame/one export,
            # per-QSA selected bits, host bookkeeping and pre-gate readback.
            half_stride=2*2560+2*4+24*256+5*2*256+8
            # At most one layer replaces a bucket at once under Model's lock;
            # the new HostRecipe exists before the previous one is destroyed.
            half_host=2*48*half_stride*4+2*48+4096+12*2*256*2+13*(24*256*4+16384)
            half_pinned=13*24*256*4
            half_hbm=13*(24*256*4+4*(1<<20))
            assert half_plan['half_capture_host_bytes']==half_host and half_plan['half_capture_pinned_bytes']==half_pinned
            assert half_plan['half_capture_hbm_bytes']==half_hbm and half_plan['dump_host_bytes']==half_host
            assert half_plan['host_required']-shorter['host_required']==half_host and half_plan['control_bytes']-shorter['control_bytes']==half_host
            assert half_plan['hbm_required']-shorter['hbm_required']==half_hbm
            assert half_plan['no_device_acquired'] and not half_file.exists()
            assert all(shorter[key]==0 for key in ['half_capture_host_bytes','half_capture_hbm_bytes','half_capture_pinned_bytes'])
            for resource in ['host','hbm']:
                limit=(half_plan[resource+'_required']-1)//(1<<20)
                refused_half=subprocess.run(external_command+half_flags+[f'--{resource}-limit-mib',str(limit)],capture_output=True,text=True)
                assert refused_half.returncode and 'memory policy' in refused_half.stderr and not half_file.exists()
            pinned_refusal=subprocess.run(external_command+half_flags+['--verify-window','1','--pinned-limit-mib','1'],capture_output=True,text=True)
            assert pinned_refusal.returncode and 'half capture and verify head exceed pinned memory policy' in pinned_refusal.stderr and not half_file.exists()
            assert subprocess.run(external_command+['--verify-window','1','--pinned-limit-mib','1'],capture_output=True,text=True).returncode==0
            # No --plan: incompatible real requests must fail BEFORE Model
            # construction; executing the compatible configuration needs HPU.
            actual=[x for x in external_command if x!='--plan-json']+half_flags+['--tokens','1,2']
            for flags,message in [([], 'prompt cache'),(['--prompt-cache','0','--prefill','2','--dump-logits',str(layer_file)],'drop --prefill'),(['--prompt-cache','0','--suffix-draft','3','--spec','2'],'drop speculative drafts')]:
                refused_half=subprocess.run(actual+flags,capture_output=True,text=True)
                assert refused_half.returncode and message in refused_half.stderr, refused_half.stdout+refused_half.stderr
                assert not half_file.exists()
            # Compatible raw diagnostics must reach actual Model preflight,
            # rather than the former blanket --prefill refusal. A 1MiB HBM
            # cap necessarily fails this full model BEFORE device acquisition.
            for prefill in [['--prefill','2'],['--prefill','auto'],['--prefill','2','--prefill-reference']]:
                admitted_raw=subprocess.run(actual+['--prompt-cache','0','--dump-layers',str(layer_file),'--hbm-limit-mib','1']+prefill,capture_output=True,text=True)
                assert admitted_raw.returncode and 'memory policy' in admitted_raw.stderr and 'drop --prefill' not in admitted_raw.stderr
                assert not half_file.exists() and not layer_file.exists()
            for flag in ['--dump-logits','--dump-layers','--dump-halves','--dump-final-r','--dump-mixed','--dump-residual']:
                resident_refusal=subprocess.run(external_command+['--serve',flag,str(layer_file)],capture_output=True,text=True)
                assert resident_refusal.returncode and 'only --dump-routing' in resident_refusal.stderr
                assert not layer_file.exists()
            layered=json.loads(subprocess.run(external_command+['--dump-layers',str(layer_file)],capture_output=True,text=True,check=True).stdout)
            ladder=(48+1)*4*2560*4
            assert layered['dump_host_bytes']==ladder and layered['host_required']-shorter['host_required']==ladder and not layer_file.exists()
            window_command=external_command+['--verify-window','7']
            window_plain=json.loads(subprocess.run(window_command,capture_output=True,text=True,check=True).stdout)
            window_routed=json.loads(subprocess.run(window_command+['--dump-routing',str(route_file),'--dump-layers',str(layer_file)],capture_output=True,text=True,check=True).stdout)
            # The minimum capture envelope is seven rows even without native
            # verify: scalar reference windows assemble only IDs in its export
            # scratch, while one scalar routing export is briefly alive.
            window_charge=charge
            assert window_routed['routing_capture_host_bytes']==window_charge
            assert window_routed['dump_host_bytes']==window_charge+ladder
            assert window_routed['host_required']-window_plain['host_required']==window_charge+ladder
            assert window_routed['hbm_required']==window_plain['hbm_required'] and not route_file.exists() and not layer_file.exists()
            host_mib=(window_routed['host_required']-1)//(1<<20)
            rejected=subprocess.run(window_command+['--dump-routing',str(route_file),'--dump-layers',str(layer_file),'--host-limit-mib',str(host_mib)],capture_output=True,text=True)
            assert rejected.returncode and 'host memory policy' in rejected.stderr and not route_file.exists() and not layer_file.exists()
            # Real Model::preflight must price fixed ladder + row-dependent
            # routing capture DURING auto selection, not reject a base-only
            # maximum after the fact. check_auto exercises tight host/HBM
            # limits and independently refuses every next larger explicit row.
            diagnostics=['--dump-routing',str(route_file),'--dump-layers',str(layer_file)]
            check_auto(external_command+diagnostics)
            check_auto(window_command+diagnostics)
            combined=json.loads(subprocess.run(external_command+diagnostics+half_flags,capture_output=True,text=True,check=True).stdout)
            assert combined['dump_host_bytes']==charge+ladder+half_host
            assert combined['host_required']-shorter['host_required']==charge+ladder+half_host
            assert combined['hbm_required']-shorter['hbm_required']==half_hbm
            check_auto(external_command+diagnostics+half_flags)
            check_auto(window_command+diagnostics+half_flags)
            assert not half_file.exists()
            # Below the smallest legal verify window, fail before creating
            # either file; auto must not turn requested verify/diagnostics off.
            refused_auto=subprocess.run(window_command+diagnostics+['--prefill','auto','--host-limit-mib',str(host_mib)],capture_output=True,text=True)
            assert refused_auto.returncode and 'memory policy' in refused_auto.stderr
            assert not route_file.exists() and not layer_file.exists()
            # A serve configuration can be planned without starting the server
            # or acquiring a device. Slot token histories are not free RAM.
            batch_plan=json.loads(subprocess.run(external_command+['--serve','--batch','2'],capture_output=True,text=True,check=True).stdout)
            assert batch_plan['slot_host_bytes']==2*(128*4+8192)
            assert batch_plan['host_required']-shorter['host_required']==batch_plan['slot_host_bytes']
            assert batch_plan['control_bytes']-shorter['control_bytes']==batch_plan['slot_host_bytes']
            assert batch_plan['hbm_required']==shorter['hbm_required'] and batch_plan['no_device_acquired']
            batch_routing=json.loads(subprocess.run(external_command+['--serve','--batch','2','--dump-routing',str(route_file)],capture_output=True,text=True,check=True).stdout)
            assert batch_routing['host_required']==batch_plan['host_required']+charge and batch_routing['dump_host_bytes']==charge
            assert batch_routing['hbm_required']==batch_plan['hbm_required'] and batch_routing['no_device_acquired'] and not route_file.exists()
            # Images have their own hard snapshot pool; bank/interrupted-read
            # vectors are EXTRA RAM, including the supported 64-entry maximum.
            assert batch_plan['cache_metadata_bytes']==64*(128*12+248320*8+4096)
            assert batch_plan['scalar_window_bytes']==(128+248320)*4+4096
            # Scalar draft rollback admission is checked BEFORE native Model
            # construction; no device is acquired for these CLI plan cases.
            no_snap=external_command+['--snapshot-budget-mib','0']
            assert json.loads(subprocess.run(no_snap,capture_output=True,text=True,check=True).stdout)['no_device_acquired']
            assert subprocess.run(no_snap+['--spec','1','--suffix-draft','3'],capture_output=True,text=True).returncode==0
            for source in [['--spec','2','--suffix-draft','3'],['--lookup-chain','2'],['--spec-oracle','unused-plan-oracle.ids']]:
                refused_scalar=subprocess.run(no_snap+source,capture_output=True,text=True)
                assert refused_scalar.returncode and 'scalar candidate verification requires nonzero --snapshot-budget-mib' in refused_scalar.stderr
            assert subprocess.run(no_snap+['--suffix-draft','3','--verify-window','2'],capture_output=True,text=True).returncode==0
            assert batch_plan['control_bytes']>=batch_plan['cache_metadata_bytes']+batch_plan['slot_host_bytes']
            old_host=batch_plan['host_required']-batch_plan['cache_metadata_bytes']
            refused=subprocess.run(external_command+['--serve','--batch','2','--host-limit-mib',str((old_host+(1<<20)-1)//(1<<20))],capture_output=True,text=True)
            assert refused.returncode and 'memory policy' in refused.stderr, refused.stdout+refused.stderr
            assert subprocess.run(external_command + ['--ple-gguf', str(small)], capture_output=True, text=True).returncode
            assert subprocess.run(external_command + ['--ple-gguf', str(external) + '.missing'], capture_output=True, text=True).returncode
            assert subprocess.run(external_command + ['--ple-gguf', ''], capture_output=True, text=True).returncode
            from native_pack_model import fixture as pack_fixture
            pack=pathlib.Path(directory)/'native expert pack'
            pack_fixture(pack,without_table)
            packed_source=json.loads(subprocess.run(external_command+['--expert-pack',str(pack)],capture_output=True,text=True,check=True).stdout)
            assert packed_source['native_expert_pack_host_bytes']==4<<20
            assert packed_source['host_required']-shorter['host_required']==4<<20
            assert packed_source['hbm_required']==shorter['hbm_required']
            assert subprocess.run(external_command+['--expert-pack',str(pack)+'.missing'],capture_output=True,text=True).returncode
            assert subprocess.run(external_command+['--expert-pack',''],capture_output=True,text=True).returncode
            pack_fixture(pack,without_table,binary=True)
            assert json.loads(subprocess.run(external_command+['--expert-pack',str(pack)],capture_output=True,text=True,check=True).stdout)==packed_source
            provenance=pack/'experts.bin.src.json'
            original=provenance.read_text()
            corrupted=json.loads(original);corrupted['native_experts_sha256']='0'*64
            provenance.write_text(json.dumps(corrupted))
            assert subprocess.run(external_command+['--expert-pack',str(pack)],capture_output=True,text=True).returncode
            provenance.write_text(original)
            native = subprocess.run([sys.argv[2], '--native', str(path), '--plan'], capture_output=True, text=True, check=True)
            assert 'no_device_acquired' in native.stdout
            native_json = subprocess.run([sys.argv[2], '--native', str(path), '--plan-json'], capture_output=True, text=True, check=True)
            assert json.loads(native_json.stdout) == plan
            windows=[sys.argv[2],'--native',str(path),'--plan-json','--prefill','8','--short-read','2','--prompt-cache-tail']
            windowed=json.loads(subprocess.run(windows,capture_output=True,text=True,check=True).stdout)
            assert windowed['prefill_rows']==8 and windowed['prefill_host_bytes']>0 and windowed['prefill_hbm_bytes']>0
            assert windowed['host_required']-plan['host_required']==windowed['prefill_host_bytes']
            assert windowed['hbm_required']-plan['hbm_required']==windowed['prefill_hbm_bytes']
            assert json.loads(subprocess.run(windows+['--prefill-reference'],capture_output=True,text=True,check=True).stdout)==plan
            bounded=json.loads(subprocess.run(windows+['--prefill','32768'],capture_output=True,text=True,check=True).stdout)
            doubled=json.loads(subprocess.run(windows+['--prefill','16'],capture_output=True,text=True,check=True).stdout)
            assert bounded['prefill_rows']==128 and doubled['prefill_rows']==16
            # Beyond the GDN cap8, its shared state/graph allowance stays fixed;
            # only session/HC/expert row buffers continue growing linearly.
            for resource in ['host','hbm']:
                key='prefill_'+resource+'_bytes'
                increment=doubled[key]-windowed[key]
                assert increment>0 and bounded[key]==windowed[key]+15*increment
                assert bounded[key]<16*windowed[key]
            for resource in ['host','hbm']:
                limit=(windowed[resource+'_required']-1)//(1<<20)
                assert subprocess.run(windows+[f'--{resource}-limit-mib',str(limit)],capture_output=True,text=True).returncode
            check_auto([sys.argv[2],'--native',str(path)])
            candidate_command=[sys.argv[2],'--native',str(path),'--max-context','128','--verify-window','7']
            candidates=json.loads(subprocess.run(candidate_command+['--plan-json'],capture_output=True,text=True,check=True).stdout)
            assert candidates['prefill_rows']==0 and candidates['window_rows']==7 and candidates['verify_rows']==7
            assert candidates['verify_logits_bytes']==7*248320*4
            assert candidates['verify_host_bytes']==25*248320*4+128*4
            assert candidates['verify_head_pinned_bytes']==7*(2560*2+248320*4)
            assert candidates['verify_hbm_bytes']==candidates['verify_head_pinned_bytes']+(64<<20)
            assert subprocess.run(candidate_command+['--plan-json','--pinned-limit-mib','6'],capture_output=True,text=True).returncode
            assert candidates['host_required']-shorter['host_required']==candidates['prefill_host_bytes']+candidates['verify_host_bytes']
            assert candidates['hbm_required']-shorter['hbm_required']==candidates['prefill_hbm_bytes']+candidates['verify_hbm_bytes']
            both=json.loads(subprocess.run(candidate_command+['--plan-json','--prefill','128'],capture_output=True,text=True,check=True).stdout)
            trunk128=json.loads(subprocess.run(candidate_command+['--plan-json','--verify-window','0','--prefill','128'],capture_output=True,text=True,check=True).stdout)
            assert both['hbm_required']-trunk128['hbm_required']==both['verify_hbm_bytes'] and both['window_rows']==128
            assert both['host_required']-trunk128['host_required']==both['verify_host_bytes']
            check_auto(candidate_command)
            for extra in [['--prefill','32769'],['--prefill','AUTO'],['--verify-window','8'],['--verify-window','-1'],['--short-read','2147483648'],['--short-read','-1']]:
                assert subprocess.run(windows+extra,capture_output=True,text=True).returncode
            sys.dont_write_bytecode = True
            from control_vector import fixture as control_fixture
            control = pathlib.Path(directory) / 'control vector.gguf'
            control_fixture(control, hidden=2560)
            control_command = [sys.argv[2], '--native', str(path), '--plan-json', '--max-context', '128', '--control-vector-scaled', f'{control}:1', '--cvec-mode', 'project', '--cvec-dir', 'single:1', '--control-vector-layer-range', '4', '44']
            controlled = json.loads(subprocess.run(control_command, capture_output=True, text=True, check=True).stdout)
            host_extra = 2 * (48 * 2560 + 48) * 4 + (1 << 20)
            hbm_extra = (2560 * (4 * 3 + 1) + 4 * 2 + 1) * 4 + (1 << 20)
            assert controlled['control_vector_host_bytes'] == host_extra
            assert controlled['control_vector_hbm_bytes'] == hbm_extra
            assert controlled['host_required'] - shorter['host_required'] == host_extra
            assert controlled['hbm_required'] - shorter['hbm_required'] == hbm_extra
            assert controlled['dense_bytes'] == shorter['dense_bytes']
            for extra in [['--cvec-dir', 'single:0'], ['--control-vector-scaled', f'{control}:inf'], ['--control-vector', str(control) + '.missing']]:
                invalid_control = subprocess.run(control_command + extra, capture_output=True, text=True)
                assert invalid_control.returncode and 'no_device_acquired' not in invalid_control.stdout
            native = subprocess.run([sys.argv[2], '--native', str(path), '--plan', '--hbm-limit-mib', '1'], capture_output=True, text=True)
            assert native.returncode and 'memory policy' in native.stderr
            sys.dont_write_bytecode = True
            from mtp_weights import fixture as mtp_fixture
            mtp = pathlib.Path(directory) / 'mtp'
            mtp.mkdir()
            mtp_fixture(mtp)
            command = [sys.argv[2], '--native', str(path), '--plan', '--max-context', '128', '--mtp', str(mtp), '--spec', '4']
            native = subprocess.run(command, capture_output=True, text=True, check=True)
            assert 'no_device_acquired' in native.stdout
            numbers = dict(field.split('=') for field in native.stdout.split() if '=' in field)
            assert int(numbers['dense_bytes']) > shorter['dense_bytes']
            subset = subprocess.run(command + ['--mtp-draft-vocab', str(mtp / 'subset_valid.bin')], capture_output=True, text=True, check=True)
            subset_numbers = dict(field.split('=') for field in subset.stdout.split() if '=' in field)
            assert int(subset_numbers['dense_bytes']) - int(numbers['dense_bytes']) == 3 * 2560 * 2
            (mtp / 'draft_vocab.bin').write_bytes((mtp / 'subset_valid.bin').read_bytes())
            automatic = subprocess.run(command, capture_output=True, text=True, check=True)
            assert automatic.stdout == subset.stdout
            for extra in [['--mtp-window', '129'], ['--mtp-expert-cache-mib', '1'], ['--spec-min-p', 'nan']]:
                invalid = subprocess.run(command + extra, capture_output=True, text=True)
                assert invalid.returncode, extra
            native_json = subprocess.run(command + ['--plan-json'], capture_output=True, text=True, check=True)
            mtp_plan = json.loads(native_json.stdout)
            assert mtp_plan['mtp_dense_bytes'] and mtp_plan['mtp_state_bytes'] and mtp_plan['mtp_cache_bytes']
            assert mtp_plan['packed_expert_execution_bytes'] == plan['packed_expert_execution_bytes'] // 48 * 49
            assert mtp_plan['no_device_acquired'] and mtp_plan['layers'] == 48
            assert not mtp_plan['mtp_q4_projections'] and not mtp_plan['mtp_q4_head']
            assert mtp_plan['mtp_window_rows']==0 and mtp_plan['mtp_window_hbm_bytes']==0
            batch_command=command+['--serve','--batch','2','--plan-json']
            batch_mtp=json.loads(subprocess.run(batch_command+['--batch-mtp'],capture_output=True,text=True,check=True).stdout)
            assert batch_mtp['no_device_acquired'] and batch_mtp['verify_rows']==2
            assert batch_mtp['verify_logits_bytes']==2*248320*4 and batch_mtp['slot_host_bytes']==2*(128*4+8192)
            assert batch_mtp['host_required']>=mtp_plan['host_required']+batch_mtp['slot_host_bytes']
            assert batch_mtp['hbm_required']>mtp_plan['hbm_required'] and batch_mtp['mtp_window_rows']==2
            env=dict(os.environ,STRATA_BATCH_MTP='1')
            assert json.loads(subprocess.run(batch_command,env=env,capture_output=True,text=True,check=True).stdout)==batch_mtp
            assert json.loads(subprocess.run(batch_command+['--batch-mtp','--verify-window','1'],capture_output=True,text=True,check=True).stdout)==batch_mtp
            plain=json.loads(subprocess.run(batch_command,capture_output=True,text=True,check=True).stdout)
            assert plain['verify_rows']==0 and plain['mtp_window_rows']==0
            mtp_window=json.loads(subprocess.run(command+['--plan-json','--prefill','17'],capture_output=True,text=True,check=True).stdout)
            assert mtp_window['mtp_window_rows']==8 and mtp_window['mtp_window_pinned_bytes']>0
            assert mtp_window['hbm_required']-mtp_plan['hbm_required']==mtp_window['prefill_hbm_bytes']+mtp_window['mtp_window_hbm_bytes']
            assert mtp_window['host_required']-mtp_plan['host_required']==mtp_window['prefill_host_bytes']+mtp_window['mtp_window_host_bytes']
            narrow=json.loads(subprocess.run(command+['--plan-json','--prefill','17','--mtp-window','1'],capture_output=True,text=True,check=True).stdout)
            assert narrow['mtp_window_hbm_bytes']<mtp_window['mtp_window_hbm_bytes']
            assert narrow['mtp_window_host_bytes']<mtp_window['mtp_window_host_bytes']
            pinned_mib=(mtp_window['mtp_window_pinned_bytes']-1)//(1<<20)
            assert subprocess.run(command+['--plan-json','--prefill','17','--pinned-limit-mib',str(pinned_mib)],capture_output=True,text=True).returncode
            # A combined plan catches auto admission before optional extras,
            # including a separate pack's provenance and control-vector bytes.
            check_auto(command+['--native',str(without_table),'--ple-gguf',str(external),'--control-vector',str(control),'--expert-pack',str(pack)])
            check_auto(command+['--native',str(without_table),'--ple-gguf',str(external),'--control-vector',str(control),'--expert-pack',str(pack),'--verify-window','7'])
            check_auto(command+['--native',str(without_table),'--ple-gguf',str(external),'--control-vector',str(control),'--expert-pack',str(pack),'--verify-window','7']+diagnostics)
            check_auto(command+['--native',str(without_table),'--ple-gguf',str(external),'--control-vector',str(control),'--expert-pack',str(pack),'--verify-window','7']+diagnostics+half_flags)
            projection_saving = (2 * 2560 * 2560 + 2560 * (12288 + 512 + 512 + 6144)) // 32 * 16
            for mode in ['proj', 'head', 'all']:
                q4 = json.loads(subprocess.run(command + ['--plan-json', '--mtp-q4', mode], capture_output=True, text=True, check=True).stdout)
                expected = mtp_plan['mtp_dense_bytes']
                if mode != 'head':
                    expected -= projection_saving
                if mode != 'proj':
                    expected += 3 * (2560 // 32 * 18 - 2560 * 2)
                assert q4['mtp_dense_bytes'] == expected
                assert q4['hbm_required'] - mtp_plan['hbm_required'] == expected - mtp_plan['mtp_dense_bytes']
                assert q4['host_required'] == mtp_plan['host_required']
                assert q4['mtp_q4_projections'] == (mode != 'head')
                assert q4['mtp_q4_head'] == (mode != 'proj')
            for extra in [['--mtp-q4', 'invalid'], ['--mtp-q4', '']]:
                assert subprocess.run(command + extra, capture_output=True, text=True).returncode
            assert subprocess.run([sys.argv[2], '--native', str(path), '--plan-json', '--mtp-q4', 'all'], capture_output=True, text=True).returncode
        for kwargs, expected in [({'missing': 'blk.47.indexer.k_norm.weight'}, 'not found'),
                                 ({'bad_shape': True}, 'shape mismatch')]:
            fixture(path, **kwargs)
            failure = subprocess.run([executable, str(path)], capture_output=True, text=True)
            assert failure.returncode and expected in failure.stderr, failure.stderr
        for kind, block, size in [(2,32,18),(3,32,20),(6,32,22),(7,32,24),(8,32,34),(42,64,18),(11,256,110),(12,256,144),(13,256,176),(14,256,210),(20,32,18),(23,256,136),(16,256,66),(17,256,74),(18,256,98),(21,256,110),(22,256,82),(29,256,56)]:
            fixture(path, head_type=kind)
            packed = json.loads(subprocess.run([executable, str(path)], capture_output=True, text=True, check=True).stdout)
            difference = 2560 * 248320 * 2 - 2560 // block * size * 248320
            assert plan['head_bytes'] - packed['head_bytes'] == difference
            assert plan['dense_bytes'] - packed['dense_bytes'] == difference
            assert plan['hbm_required'] - packed['hbm_required'] == difference
            assert packed['host_required'] == plan['host_required']
            if len(sys.argv) > 2:
                base_command = [sys.argv[2], '--native', str(path), '--plan-json', '--max-context', '128', '--mtp', str(mtp), '--spec', '4']
                # Automatic subset and explicit full/shared head use identical
                # packed row sizes, not a hidden BF16 subset allocation.
                (mtp / 'draft_vocab.bin').unlink(missing_ok=True)
                whole = json.loads(subprocess.run(base_command, capture_output=True, text=True, check=True).stdout)
                subset = json.loads(subprocess.run(base_command + ['--mtp-draft-vocab', str(mtp / 'subset_valid.bin')], capture_output=True, text=True, check=True).stdout)
                assert subset['mtp_dense_bytes'] - whole['mtp_dense_bytes'] == 3 * (2560 // block * size)
                assert subset['hbm_required'] - whole['hbm_required'] == 3 * (2560 // block * size)
                q4_whole = json.loads(subprocess.run(base_command + ['--mtp-q4', 'head'], capture_output=True, text=True, check=True).stdout)
                q4_subset = json.loads(subprocess.run(base_command + ['--mtp-q4', 'head', '--mtp-draft-vocab', str(mtp / 'subset_valid.bin')], capture_output=True, text=True, check=True).stdout)
                assert q4_whole['mtp_dense_bytes'] - whole['mtp_dense_bytes'] == 248320 * (2560 // 32 * 18)
                assert q4_subset['mtp_dense_bytes'] - q4_whole['mtp_dense_bytes'] == (3 - 248320) * (2560 // 32 * 18)
                assert q4_whole['head_bytes'] == whole['head_bytes']  # target head keeps its original arithmetic
        if len(sys.argv)>2:
            fixture(path,expert_type=20)
            env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_HPU_')}
            command=[sys.argv[2],'--native',str(path),'--plan-json','--fp8-experts','--expert-cache-mib','2048',
                     '--max-context','128','--hbm-limit-mib','98304']
            plain_run=subprocess.run(command,capture_output=True,text=True,env=env,check=True)
            plain=json.loads(plain_run.stdout)
            assert 'resident_iq4_bytes' not in plain
            env.update({'STRATA_HPU_RESIDENT_IQ4':'1','STRATA_HPU_CHAIN':'1'})
            resident=json.loads(subprocess.run(command,capture_output=True,text=True,env=env,check=True).stdout)
            # Independent allocation sum: all 512 routed experts plus shared, 48 layers, one 256-byte codebook.
            payload=48*513*(1280*2560//2+1280*80*4+2560*640//2+2560*20*4)+256
            assert resident['resident_iq4_bytes']==payload==75644928256
            assert resident['resident_iq4_staging_bytes']==64*(1280*2560//2+1280*80*4)==131072000
            assert resident['hbm_required']-plain['hbm_required']==payload
            assert resident['host_required']==plain['host_required'] and resident['dense_bytes']==plain['dense_bytes']
            lower=(resident['hbm_required']-1)//(1<<20)
            rejected=subprocess.run(command+['--hbm-limit-mib',str(lower)],capture_output=True,text=True,env=env)
            assert rejected.returncode and 'resident IQ4' in rejected.stderr and 'HBM memory policy' in rejected.stderr
            upper=(resident['hbm_required']+(1<<20)-1)//(1<<20)
            assert json.loads(subprocess.run(command+['--hbm-limit-mib',str(upper)],capture_output=True,text=True,env=env,check=True).stdout)==resident
            rejected=subprocess.run(command+['--pinned-limit-mib','124'],capture_output=True,text=True,env=env)
            assert rejected.returncode and 'resident IQ4 upload staging' in rejected.stderr
            # RESIDENT_U4 swaps the IQ4 codes for the six-region U4 store the MME batch GEMM reads: same 512
            # routed experts plus the shared tail block, gate/up K=512 (5 groups) and down K=320 (2), no codebook.
            # The payload is the store's own bytes(): 513 blocks of 2,549,760 bytes per layer, and the pinned
            # staging holds eight expert blocks per upload round.
            u4_env=dict(env,STRATA_HPU_RESIDENT_U4='1')
            u4=json.loads(subprocess.run(command,capture_output=True,text=True,env=u4_env,check=True).stdout)
            block=1280*2560//2+2*1280*5*4+2560*640//2+2*2560*2*4
            assert block==2549760
            assert u4['resident_u4_bytes']==48*513*block==62785290240
            assert u4['resident_u4_staging_bytes']==8*block
            assert 'resident_iq4_bytes' not in u4
            assert u4['hbm_required']-plain['hbm_required']==62785290240
            assert u4['host_required']==plain['host_required'] and u4['dense_bytes']==plain['dense_bytes']
            assert u4['hbm_required']<resident['hbm_required']  # 62.8 vs 75.6 GB: U4 drops the IQ4 block scales
            lower=(u4['hbm_required']-1)//(1<<20)
            rejected=subprocess.run(command+['--hbm-limit-mib',str(lower)],capture_output=True,text=True,env=u4_env)
            assert rejected.returncode and 'resident U4' in rejected.stderr and 'HBM memory policy' in rejected.stderr
            upper=(u4['hbm_required']+(1<<20)-1)//(1<<20)
            assert json.loads(subprocess.run(command+['--hbm-limit-mib',str(upper)],capture_output=True,text=True,env=u4_env,check=True).stdout)==u4
            # The prefill window pool is priced independently of the expert store, so its reservation delta holds.
            u4_window=json.loads(subprocess.run(command+['--prefill','8'],capture_output=True,text=True,env=dict(u4_env,STRATA_HPU_QSA_DEVICE='1'),check=True).stdout)
            assert u4_window['hbm_required']-u4['hbm_required']==u4_window['prefill_hbm_bytes']
            assert u4_window['host_required']-u4['host_required']==u4_window['prefill_host_bytes']
            # --prefill 2..8 is the resident window: one verify chain (shared with a DFlash round), never the
            # grouped per-layer window pool. Its reservation replaces that pool's, so --plan stays exact.
            prefill_env=dict(env);prefill_env['STRATA_HPU_QSA_DEVICE']='1'
            windowed=json.loads(subprocess.run(command+['--prefill','8'],capture_output=True,text=True,env=prefill_env,check=True).stdout)
            assert windowed['prefill_rows']==8 and windowed['window_rows']==0
            assert windowed['prefill_host_bytes']>0 and windowed['prefill_hbm_bytes']>0
            assert windowed['host_required']-resident['host_required']==windowed['prefill_host_bytes']
            assert windowed['hbm_required']-resident['hbm_required']==windowed['prefill_hbm_bytes']
            rejected=subprocess.run(command+['--prefill','9'],capture_output=True,text=True,env=prefill_env)
            assert rejected.returncode and 'prefill >8 requires STRATA_HPU_Q8_PACKED' in rejected.stderr
            wide_env=dict(prefill_env,STRATA_HPU_Q8_PACKED='1')
            for rows in (9,32,64,128):
                wide=json.loads(subprocess.run(command+['--prefill',str(rows)],capture_output=True,text=True,env=wide_env,check=True).stdout)
                assert wide['prefill_rows']==rows and wide['window_rows']==0
                assert wide['host_required']-resident['host_required']==wide['prefill_host_bytes']
                assert wide['hbm_required']-resident['hbm_required']==wide['prefill_hbm_bytes']
                low_hbm=(wide['hbm_required']-1)//(1<<20)
                refused=subprocess.run(command+['--prefill',str(rows),'--hbm-limit-mib',str(low_hbm)],capture_output=True,text=True,env=wide_env)
                assert refused.returncode and 'memory policy' in refused.stderr
            # Final-state prefill retains one state, rather than each row's state.
            # Baseline runtime reserve already covers eight pairs, so only the extra rows disappear.
            final_env=dict(wide_env,STRATA_HPU_PREFILL_FINAL='1')
            for rows in (32,64,128):
                original=json.loads(subprocess.run(command+['--prefill',str(rows)],capture_output=True,text=True,env=wide_env,check=True).stdout)
                final=json.loads(subprocess.run(command+['--prefill',str(rows)],capture_output=True,text=True,env=final_env,check=True).stdout)
                saved=36*(rows-8)*(128*128*48+3*10240)*4
                assert original['hbm_required']-final['hbm_required']==saved
                assert original['host_required']==final['host_required']
                low=(final['hbm_required']-1)//(1<<20)
                refused=subprocess.run(command+['--prefill',str(rows),'--hbm-limit-mib',str(low)],capture_output=True,text=True,env=final_env)
                assert refused.returncode and 'memory policy' in refused.stderr
            canonical_env=dict(final_env,STRATA_HPU_PREFILL_QSA32='1')
            for rows in (64,128):
                base=json.loads(subprocess.run(command+['--prefill',str(rows)],capture_output=True,text=True,env=final_env,check=True).stdout)
                canonical=json.loads(subprocess.run(command+['--prefill',str(rows)],capture_output=True,text=True,env=canonical_env,check=True).stdout)
                assert canonical['hbm_required']-base['hbm_required']==128+(128 if rows==128 else 64)*(1<<20)
                assert canonical['host_required']-base['host_required']==128+(128 if rows==128 else 64)*(1<<20)
            for rows in (8,32,96):
                refused=subprocess.run(command+['--prefill',str(rows)],capture_output=True,text=True,env=canonical_env)
                assert refused.returncode and 'PREFILL_QSA32 requires' in refused.stderr
            refused=subprocess.run(command+['--prefill','64'],capture_output=True,text=True,env=dict(wide_env,STRATA_HPU_PREFILL_QSA32='1'))
            assert refused.returncode and 'PREFILL_QSA32 requires' in refused.stderr
            # Chunk attention independently of the full FFN window. The original
            # plan remains exact when explicitly disabled. Check the reduced HBM
            # reservation against the independent per-row KV/query/mask geometry.
            gather_env=dict(canonical_env,STRATA_HPU_QSA_DEVICE_GATHER='1')
            for rows in (64,128):
                args=command+['--prefill',str(rows)]
                base=json.loads(subprocess.run(args,capture_output=True,text=True,env=gather_env,check=True).stdout)
                off=json.loads(subprocess.run(args,capture_output=True,text=True,env=dict(gather_env,STRATA_HPU_QSA_CHUNK_ROWS='0'),check=True).stdout)
                assert off==base
                for chunk in (16,32):
                    chunk_env=dict(gather_env,STRATA_HPU_QSA_CHUNK_ROWS=str(chunk))
                    chunk_plan=json.loads(subprocess.run(args,capture_output=True,text=True,env=chunk_env,check=True).stdout)
                    # At max-context 128 every window stays inside the device cache's identity selection (context + 1
                    # <= top_k + block - 1), so the staged gather window is unreachable and not priced (model.cpp,
                    # staged_reachable): chunking its attention cannot change the plan.
                    assert base['hbm_required']==chunk_plan['hbm_required']
                    assert chunk_plan['prefill_rows']==rows and chunk_plan['host_required']==base['host_required']
                    low=(chunk_plan['hbm_required']-1)//(1<<20)
                    refused=subprocess.run(args+['--hbm-limit-mib',str(low)],capture_output=True,text=True,env=chunk_env)
                    assert refused.returncode and 'memory policy' in refused.stderr
            for value in ('8','17','32x','-1'):
                refused=subprocess.run(command+['--prefill','64'],capture_output=True,text=True,env=dict(gather_env,STRATA_HPU_QSA_CHUNK_ROWS=value))
                assert refused.returncode and 'QSA_CHUNK_ROWS must' in refused.stderr
            refused=subprocess.run(command+['--prefill','64'],capture_output=True,text=True,env=dict(canonical_env,STRATA_HPU_QSA_CHUNK_ROWS='32'))
            assert refused.returncode and 'QSA_CHUNK_ROWS requires' in refused.stderr
            # CLI prompt chunks can be larger; native model windows cap at128.
            capped=json.loads(subprocess.run(command+['--prefill','129'],capture_output=True,text=True,env=wide_env,check=True).stdout)
            assert capped['prefill_rows']==128 and capped['window_rows']==0
            rejected=subprocess.run(command+['--prefill','8'],capture_output=True,text=True,env=env)
            assert rejected.returncode and 'prefill window needs STRATA_HPU_QSA_DEVICE' in rejected.stderr
            env.pop('STRATA_HPU_RESIDENT_IQ4');env.pop('STRATA_HPU_CHAIN')
            assert subprocess.run(command,capture_output=True,text=True,env=env,check=True).stdout==plain_run.stdout
            # Canonical FP32 embedding RAM is host-only and must be admitted before device acquisition.
            env['STRATA_HPU_EMBED_RAM']='1'
            cached=json.loads(subprocess.run(command,capture_output=True,text=True,env=env,check=True).stdout)
            embedding_charge=248320*2560*4+65536
            assert cached['embedding_ram_host_bytes']==embedding_charge
            assert cached['host_required']-plain['host_required']==embedding_charge
            assert cached['hbm_required']==plain['hbm_required'] and cached['dense_bytes']==plain['dense_bytes']
            host_floor=(cached['host_required']-1)//(1<<20)
            rejected=subprocess.run(command+['--host-limit-mib',str(host_floor)],capture_output=True,text=True,env=env)
            assert rejected.returncode and 'memory policy' in rejected.stderr
            host_ceil=(cached['host_required']+(1<<20)-1)//(1<<20)
            assert json.loads(subprocess.run(command+['--host-limit-mib',str(host_ceil)],capture_output=True,text=True,env=env,check=True).stdout)==cached
            env.pop('STRATA_HPU_EMBED_RAM')
            assert subprocess.run(command,capture_output=True,text=True,env=env,check=True).stdout==plain_run.stdout
        nv_locations=fixture(path,nvfp4_roles=('gate','up','down'))
        nv=json.loads(subprocess.run([executable,str(path)],capture_output=True,text=True,check=True).stdout)
        reserve=(8*2560+10*640)*(1+8+32+128)
        assert nv['scaled_expert_execution_bytes']==reserve
        assert nv['hbm_required']-plan['hbm_required']==reserve
        assert nv['dense_bytes']==plan['dense_bytes']
        with path.open('r+b') as file:
            file.seek(nv_locations['blk.0.ffn_down_exps.scale']+511*4)
            file.write(struct.pack('<f',0))
        invalid=subprocess.run([executable,str(path)],capture_output=True,text=True)
        assert invalid.returncode and 'positive finite' in invalid.stderr
    print('PASS complete schema, packed/Q4 head budgets, NVFP4 scale preflight/reserve and invalid metadata rejection')


if __name__ == '__main__':
    main(sys.argv[1])
