"""Actual original GGUF parser/PLE reader, no model metadata or device needed."""
import pathlib
import json
import os
import struct
import subprocess
import sys
import tempfile


def text(value):
    encoded = value.encode()
    return struct.pack('<Q', len(encoded)) + encoded


def fixture(path, *, kind=20, rows=32, columns=160, scale=1., marker='f8_e4m3',
            extra=0, offset=0, adjacent=False):
    # Deliberately looks like shard 2/2; there is no companion shard. External
    # PLE lookup must not invoke model/split-family geometry resolution.
    metadata = [('split.no', 4, struct.pack('<I', 1)),
                ('split.count', 4, struct.pack('<I', 2))]
    if kind == 24:
        if marker is not None:
            metadata.append(('strata.ple.format', 8, text(marker)))
        metadata.append(('strata.ple.scale', 12, struct.pack('<d', scale)))
    header = struct.pack('<IIQQ', 0x46554747, 3, 2 if adjacent else 1, len(metadata))
    for key, tag, value in metadata:
        header += text(key) + struct.pack('<I', tag) + value
    header += text('per_layer_token_embd.weight') + struct.pack('<IQQIQ', 2, columns, rows, kind, offset)
    if adjacent:
        header += text('unrelated.weight') + struct.pack('<IQIQ', 1, 8, 0, 0)
    header += bytes((-len(header)) % 32)
    row_bytes = {20: 90, 6: 110, 24: 160, 7: 120, 8: 170, 2: 90, 30: 320}.get(kind, 640)
    # Independent encodings of exactly 1 in every column. Deliberately nonzero
    # payloads expose header-relative offsets and split-half nibble errors.
    half = struct.pack('<e', 1)
    row = {20: (half + b'\x88' * 16) * 5,
           6: (half + b'\xff' * 4 + b'\x11' * 16) * 5,
           7: (half + b'\0' * 6 + b'\x11' * 16) * 5,
           8: (half + b'\x01' * 32) * 5,
           2: (half + b'\x99' * 16) * 5,
           24: b'\x38' * 160, 30: b'\x80\x3f' * 160}.get(kind, bytes(640))
    assert len(row) == row_bytes
    payload = b'\x7f' * offset + row * min(rows,64)
    if extra < 0:
        payload = payload[:extra]
    else:
        payload += bytes(extra)
    with path.open('wb') as file:
        file.write(header)
        file.write(payload)


def main(executable,record_path=None):
    observations=[]
    with tempfile.TemporaryDirectory(prefix='strata-hpu-ple-source-') as directory:
        path = pathlib.Path(directory) / 'external-00002-of-00002.gguf'
        for kind in [20, 6, 24, 7, 8, 2, 30]:
            fixture(path, kind=kind)
            result = subprocess.run([executable, str(path)], capture_output=True, text=True, check=True)
            assert result.stdout.splitlines()[0] == '32'
            assert 'ple-local-file-v1|' in result.stdout
            ram = subprocess.run([executable, str(path), 'ram'], capture_output=True, text=True, check=True)
            fields = list(map(int, ram.stdout.splitlines()[2].split()))
            assert fields[1] == {20: 90, 6: 110, 24: 160, 7: 120, 8: 170, 2: 90, 30: 320}[kind]*32
            assert fields[2] >= fields[1] and fields[4] == fields[2]+fields[3]+(1<<20)
            assert fields[5] in (0,1) # lock outcome is observable, not inferred from ram being requested
            observation={'ggml_type':kind,'payload_offset':fields[0],'payload_bytes':fields[1],
                         'page_bytes':fields[2],'worker_bytes':fields[3],'host_bytes':fields[4],
                         'locked':bool(fields[5])}
            if sys.platform.startswith('linux'):
                import resource
                page=os.sysconf('SC_PAGESIZE')
                assert fields[2]==((fields[0]+fields[1]-1)//page-fields[0]//page+1)*page
                assert fields[3]>=16*(1<<20),'POSIX pretouch worker reservation missing'
                def no_lock():
                    resource.setrlimit(resource.RLIMIT_MEMLOCK,(0,0))
                fallback=subprocess.run([executable,str(path),'ram'],capture_output=True,text=True,check=True,preexec_fn=no_lock)
                # The validation account is unprivileged. Even with lock refused,
                # every row must still decode via the original mmap fallback.
                assert fallback.stdout.splitlines()[2].split()[-1]=='0',fallback.stdout
                assert 'mlock failed' in fallback.stderr,fallback.stderr
                observation['forced_lock_failure_stderr']=fallback.stderr
                observation['forced_lock_failure_rows_passed']=True
            elif sys.platform=='win32':
                assert fields[3]==0 # original Windows implementation has no touch-worker pool
            observations.append(observation)
            direct=subprocess.run([executable,str(path),'direct'],capture_output=True,text=True,check=True)
            direct_fields=list(map(float,direct.stdout.splitlines()[3].split()))
            assert direct_fields[0]==4096 and direct_fields[2]==1024*16384
            assert direct_fields[1]==4096*(fields[1]//32+4)+4096//8
            assert direct_fields[4]==sum(direct_fields[1:4])+(4<<20)
            assert direct_fields[5]>=10000,'delay did not reach both real direct reads'
            observation['direct']={'cache_rows':int(direct_fields[0]),'cache_bytes':int(direct_fields[1]),
                'inflight_bytes':int(direct_fields[2]),'worker_bytes':int(direct_fields[3]),
                'host_bytes':int(direct_fields[4]),'two_reads_us':direct_fields[5]}
            if kind==20 and sys.platform.startswith('linux'):
                environment=dict(os.environ,STRATA_IO_THREADS='64')
                maximum=subprocess.run([executable,str(path),'direct'],env=environment,capture_output=True,text=True,check=True)
                maximum_fields=list(map(float,maximum.stdout.splitlines()[3].split()))
                assert maximum_fields[3]>=65*(1<<20)
                assert int(maximum_fields[3])*16==fields[3]*65,'DirectFile pool or coordinator stack missing'
                observation['max_worker_pool_bytes']=int(maximum_fields[3])
        first = result.stdout
        old_stat = path.stat()
        replacement = path.with_suffix('.replacement')
        fixture(replacement, kind=30)
        os.utime(replacement, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        os.replace(replacement, path)
        replaced = subprocess.run([executable, str(path)], capture_output=True, text=True, check=True).stdout
        assert replaced != first, 'same-size/mtime file replacement lost source identity'
        fixture(path, kind=30, rows=33)
        assert subprocess.run([executable, str(path)], capture_output=True, text=True, check=True).stdout != first
        fixture(path, adjacent=True, offset=32)
        subprocess.run([executable, str(path)], capture_output=True, text=True, check=True)
        subprocess.run([executable, str(path),'ram'],capture_output=True,text=True,check=True)
        for kwargs in [dict(columns=159), dict(rows=0), dict(rows=2**32+1),
                       dict(kind=0), dict(kind=24, marker=None), dict(kind=24, marker='wrong'),
                       *[dict(kind=24, scale=value) for value in [0, -1, float('inf'), float('nan'), 1e300, 1e-300]],
                       dict(extra=1), dict(offset=32), dict(extra=-1)]:
            fixture(path, **kwargs)
            result = subprocess.run([executable, str(path)], capture_output=True, text=True)
            assert result.returncode, kwargs
        path.unlink()
        assert subprocess.run([executable, str(path)], capture_output=True, text=True).returncode
    if record_path:
        pathlib.Path(record_path).write_text(json.dumps({'formats':observations,'device_acquired':False,
            'hardware_tests_run':False,'goal_complete':False},indent=2)+'\n',encoding='utf-8')
    print('PASS seven original PLE formats, isolated split shard, exact row offsets, identity/malformed guards; real RAM reader reservation, page/worker charges and lock/fallback outcomes; no device')


if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)
