"""Real pipe I/O, production reader/generator and original server adapter."""
import pathlib
import queue
import subprocess
import sys
import threading
import tempfile


def main(executable):
    proc = subprocess.Popen([executable], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)
    lines = queue.Queue()

    def pump():
        for line in proc.stdout:
            lines.put(line.strip())
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()

    def read():
        line = lines.get(timeout=10)
        assert line is not None, 'engine exited early'
        return line

    def send(line):
        proc.stdin.write(line + '\n')
        proc.stdin.flush()

    def terminal():
        out = []
        while True:
            line = read()
            out.append(line)
            if line.startswith(('DONE ', 'ERR ')):
                return out

    try:
        assert read().startswith('INFO backend=hpu')
        assert read() == 'READY 1024 stop'
        send('GEN 3 1,2')
        out = terminal()
        assert [line for line in out if line.startswith('T ')] == ['T 3', 'T 4', 'T 5'], out
        assert out[-1].split()[1:3] == ['3', '2'] and out[-1].split()[5] == 'length'
        # Send as soon as DONE arrives, exercising the terminal/idle race.
        send('GEN 1 1,2,3,4,5')
        out = terminal()
        assert 'RESUME 4' in out and 'T 6' in out, out
        send('GEN 200 1,2')
        while not read().startswith('T '):
            pass
        send('STOP')
        out = terminal()
        assert out[-1].split()[5] == 'cancel', out
        send('GEN 1 top_p=nan 1')
        assert terminal()[-1].startswith('ERR ')
        send('GEN 1 254')
        out = terminal()
        assert 'T 255' in out and out[-1].split()[5] == 'stop', out
        send('VRAM 256')
        assert read() == 'VRAM expert_slots=2 vram_free_mib=1024 reserve_mib=256'
        send('VRAM')
        assert read().endswith('reserve_mib=512')
        send('VRAM -1')
        assert read().startswith('ERR ')
        with tempfile.TemporaryDirectory(prefix='strata-hpu-protocol-state-') as directory:
            path = str(pathlib.Path(directory) / 'session with spaces.bin')
            send('SAVE ' + path)
            assert read().startswith('SWAIT ')
            saved = read()
            assert saved.startswith('SAVED 1 '), saved
            send('GEN 1 1')
            terminal()
            send('RESTORE ' + path)
            assert read().startswith('SWAIT ')
            assert read().startswith('RESTORED 1 ')
            send('GEN 1 254')
            out = terminal()
            assert 'RESUME 1' in out and 'T 255' in out, out
            with open(path, 'r+b') as file:
                file.seek(-1, 2)
                byte = file.read(1)
                file.seek(-1, 2)
                file.write(bytes([byte[0] ^ 1]))
            send('RESTORE ' + path)
            assert read().startswith('SWAIT ')
            assert read().startswith('SERR invalid 0 ')
            send('GEN 1 254')
            assert 'RESUME 1' in terminal(), 'failed restore mutated active generation'
        send('GEN 1 ckpt=0 5,6')
        assert 'T 7' in terminal()
        send('GEN 1 ckpt=0 5,6')
        assert 'RESUME 0' in terminal(), 'ckpt=0 retained a live checkpoint'
        send('GEN 1 pin=2 0,1,7,8')
        assert 'T 9' in terminal()
        send('GEN 1 pin=2 0,1,9,10')
        out = terminal()
        assert 'RESUME 2' in out and 'T 11' in out, out
        send('GEN 6 0,1,2,3,4,5,6,7,8,9,0,1,2')
        out = terminal()
        assert [line for line in out if line.startswith('T ')] == [f'T {n}' for n in range(3, 9)], out
        assert out[-1].split()[6:8] == ['5', '5'], out
        send('QUIT')
        proc.wait(timeout=5)
        assert proc.returncode == 0, proc.stderr.read()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    print('PASS resident pipe protocol, rapid consecutive requests, cache reuse, STOP, invalid request recovery, EOS and QUIT')

    # Import the real server when its optional Python dependencies are present.
    # CMake registers this separately so a missing dependency is an explicit skip.
    if '--server-adapter' in sys.argv:
        root = pathlib.Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        from serve.server import StrataEngine
        engine = StrataEngine(str(pathlib.Path(executable).resolve()), [])
        try:
            assert engine.max_context == 1024 and engine.can_stop
            assert [token for token in engine.generate([1, 2], 3, {}, threading.Event()) if token is not None] == [3, 4, 5]
            assert engine.last['generated'] == 3 and engine.last['finish'] == 'length'
            assert engine.vram(256)['reserve_mib'] == 256
            with tempfile.TemporaryDirectory(prefix='strata-hpu-server-state-') as directory:
                path = str(pathlib.Path(directory) / 'snapshot.bin')
                saved = engine.session_file('save', path)
                restored = engine.session_file('restore', path)
                assert saved['tokens'] == restored['tokens'] == 4 and saved['bytes'] == restored['bytes']
            sampling = {'strata_checkpoint': False}
            assert list(token for token in engine.generate([5, 6], 1, sampling, threading.Event()) if token is not None) == [7]
            assert list(token for token in engine.generate([5, 6], 1, sampling, threading.Event()) if token is not None) == [7]
            assert engine.last['reused'] == 0
            sampling = {'strata_prefix': {'tokens': 2}}
            assert list(token for token in engine.generate([0, 1, 7, 8], 1, sampling, threading.Event()) if token is not None) == [9]
            assert list(token for token in engine.generate([0, 1, 9, 10], 1, sampling, threading.Event()) if token is not None) == [11]
            assert engine.last['reused'] == 2
            repeated = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2]
            assert list(token for token in engine.generate(repeated, 6, {}, threading.Event()) if token is not None) == list(range(3, 9))
            assert engine.last['drafts_accepted'] == engine.last['drafts_offered'] == 5
        finally:
            engine.close()
        print('PASS unmodified serve.server.StrataEngine -> native protocol -> generation -> streamed tokens')


if __name__ == '__main__':
    main(sys.argv[1])
