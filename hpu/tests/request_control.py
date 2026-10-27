"""Real POSIX host-fixture pipe tests; never acquires an HPU."""
import pathlib
import os
import queue
import subprocess
import sys
import threading
import time


class Pipe:
    def __init__(self, executable, batch=False, extra=()):
        self.proc = subprocess.Popen([executable] + (['--batch', '3'] if batch else []) + list(extra),
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.queue = queue.Queue()
        self.seen = []

        def pump():
            for line in self.proc.stdout:
                self.queue.put(line.rstrip('\n'))
            self.queue.put(None)

        self.pump = threading.Thread(target=pump, daemon=True)
        self.pump.start()
        self.until('READY ')

    def send(self, text):
        self.proc.stdin.write(text)
        self.proc.stdin.flush()

    def until(self, prefix, errors=False):
        deadline = time.monotonic() + 10
        while True:
            line = self.queue.get(timeout=max(.01, deadline - time.monotonic()))
            assert line is not None, ('premature fixture EOF', prefix, self.seen)
            self.seen.append(line)
            assert not line.startswith('FATAL '), line
            assert errors or not line.startswith('ERR '), line
            if line.startswith(prefix):
                return line

    def finish(self):
        assert self.proc.wait(timeout=10) == 0, self.proc.stderr.read()
        self.pump.join(timeout=2)
        assert not self.pump.is_alive()
        while not self.queue.empty():
            line = self.queue.get_nowait()
            if line is not None:
                self.seen.append(line)
        return self.seen

    def close(self):
        if self.proc.poll() is None:
            if not self.proc.stdin.closed:
                self.send('STOP\nQUIT\n')
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()  # ONLY our owned no-device host test fixture
                self.proc.wait(timeout=5)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            stream.close()


def frames(lines, prefix):
    return [line.split() for line in lines if line.startswith(prefix)]


def fifo_quit_eof(executable):
    for batch in (False, True):
        for eof in (False, True):
            p = Pipe(executable, batch)
            try:
                p.send('GEN 2 temperature=0 1,2\nGEN 1 temperature=0 7,8\n')
                if eof:
                    p.proc.stdin.close()
                else:
                    p.send('QUIT\nGEN 1 temperature=0 20,21\n')
                lines = p.finish()
                assert [int(f[1]) for f in frames(lines, 'T ')] == [3, 4, 9], lines
                done = frames(lines, 'DONE ')
                assert [(d[1], d[5]) for d in done] == [('2', 'length'), ('1', 'length')], lines
                assert all(len(d) >= 16 and int(d[14]) == 2 for d in done), done
                assert not frames(lines, 'ERR '), lines
            finally:
                p.close()


def eof_while_active(executable):
    # EOF must drain the running request. Closing before the worker binds its
    # active flag cannot expose accidental EOF-driven cancellation.
    for batch in (False, True):
        p = Pipe(executable, batch)
        try:
            p.send('GEN 20 temperature=0 1,2\n')
            p.until('T ')
            p.proc.stdin.close()
            lines = p.finish()
            assert [int(f[1]) for f in frames(lines, 'T ')] == list(range(3, 23)), lines
            done = frames(lines, 'DONE ')
            assert len(done) == 1 and (done[0][1], done[0][5]) == ('20', 'length'), lines
            assert not frames(lines, 'ERR '), lines
        finally:
            p.close()


def ordinary_stop(executable):
    for batch in (False, True):
        p = Pipe(executable, batch)
        try:
            prompt = ','.join(map(str, range(100)))
            p.send('GEN 8 temperature=0 ' + prompt + '\nGEN 1 temperature=0 140,141\nQUIT\n')
            p.until('PP ')
            # QUIT is already queued, but STOP still reaches the current read.
            p.send('STOP\n')
            lines = p.finish()
            done = frames(lines, 'DONE ')
            assert len(done) == 2 and done[0][1] == '0' and done[0][5] == 'cancel', lines
            assert 0 < int(done[0][14]) < 100, done
            assert (done[1][1], done[1][5]) == ('1', 'length'), done
            assert [int(f[1]) for f in frames(lines, 'T ')] == [142], lines
            assert not frames(lines, 'ERR '), lines
        finally:
            p.close()


def bgen_stop_resume(executable):
    p = Pipe(executable, True)
    try:
        assert any('batch_slots=3' in s for s in p.seen), p.seen
        p.send('BGEN 1 50 temperature=0 160,161\n')
        p.until('BADM 1 1')
        prompt = ','.join(map(str, range(80)))
        p.send('BGEN 0 8 temperature=0 ' + prompt + '\n')
        p.until('PP 1 80 ')
        p.send('STOP\nBSTOP 0\n')
        p.until('BADM 0 0')
        cancelled = [d for d in frames(p.seen, 'DONE ') if d[2] == '80'][-1]
        assert cancelled[1] == '0' and cancelled[5] == 'cancel' and 0 < int(cancelled[14]) < 80, p.seen
        p.send('BGEN 0 2 temperature=0 ' + prompt + '\n')
        p.until('BDONE 0 ')
        # The independent request must finish its complete reply, never STOP.
        if not frames(p.seen, 'BDONE 1 '):
            p.until('BDONE 1 ')
        p.send('QUIT\n')
        lines = p.finish()
        done = [d for d in frames(lines, 'DONE ') if d[2] == '80']
        assert len(done) == 2 and int(done[1][8]) == int(cancelled[14]), done
        assert int(done[1][14]) == 80 - int(done[1][8]), done
        assert [(d[2], d[3]) for d in frames(lines, 'BDONE 0 ')] == [('2', 'length')], lines
        assert [(d[2], d[3]) for d in frames(lines, 'BDONE 1 ')] == [('50', 'length')], lines
        assert not frames(lines, 'ERR '), lines
    finally:
        p.close()


def stop_bstop_scopes(executable):
    p = Pipe(executable, True)
    try:
        p.send('BGEN 0 60 temperature=0 1,2\n')
        p.until('BADM 0 1')
        p.send('STOP\n')  # nobody owns ordinary control: stale, no broadcast
        first = p.until('BDONE 0 ').split()
        assert first[2:4] == ['60', 'length'], p.seen
        p.send('STOP\nBGEN 1 60 temperature=0 20,21\n')
        p.until('BADM 1 1')
        p.send('BSTOP 1\n')
        second = p.until('BDONE 1 ').split()
        assert second[3] == 'cancel' and 2 <= int(second[2]) < 60, p.seen
        prompt = ','.join(map(str, range(30, 126)))
        p.send('BGEN 2 3 temperature=0 ' + prompt + '\n')
        p.until('PP 1 96 ')
        p.send('BSTOP 2\n')  # free slot while its prompt reads; admission resets it upstream
        third = p.until('BDONE 2 ').split()
        assert third[2:4] == ['3', 'length'], p.seen
        p.send('QUIT\n')
        lines = p.finish()
        assert not frames(lines, 'ERR '), lines
    finally:
        p.close()


def invalid_then_valid(executable):
    for batch in (False, True):
        p = Pipe(executable, batch)
        try:
            p.send('GEN 0 1\nGEN 1 temperature=0 5,6\nQUIT\n')
            lines = p.finish()
            assert len(frames(lines, 'ERR ')) == 1 and len(frames(lines, 'DONE ')) == 1, lines
            assert [f[1] for f in frames(lines, 'T ')] == ['7'], lines
        finally:
            p.close()


def server_adapter(executable):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from serve.server import StrataEngine
    engine = StrataEngine(str(pathlib.Path(executable).resolve()), ['--batch', '3'])
    try:
        assert engine.batch == 3 and engine.can_stop, engine.info
        outputs = [None, None]
        errors = []
        started = threading.Event()

        def run(index, ids, n):
            try:
                out = []
                for token in engine.generate_batched(ids, n, {}, threading.Event()):
                    if token is not None:
                        out.append(token)
                        if index == 0:
                            started.set()
                outputs[index] = out
            except BaseException as exc:
                errors.append(repr(exc))
                started.set()

        first = threading.Thread(target=run, args=(0, [1, 2], 40), daemon=True)
        second = threading.Thread(target=run, args=(1, list(range(60, 84)), 16), daemon=True)
        first.start()
        assert started.wait(5), 'first original-server request never produced a token'
        second.start()
        first.join(15)
        second.join(15)
        assert not first.is_alive() and not second.is_alive() and not errors, errors
        assert outputs == [list(range(3, 43)), list(range(84, 100))], outputs
        assert engine.alive() and not any(engine.slot_busy), engine.slot_busy
        assert [id for id in engine.generate_batched([120, 121], 2, {}, threading.Event()) if id is not None] == [122, 123]
    finally:
        engine.close()
    print('PASS unmodified serve.server.StrataEngine.generate_batched: native protocol capability, concurrent control handoff, independent complete replies and subsequent reuse; host model fixture only')


def server_cleanup(executable):
    from serve.server import StrataEngine
    engine = StrataEngine(str(pathlib.Path(executable).resolve()), ['--batch', '3'])
    try:
        # Cancel during a real prompt read through the original adapter's PP
        # handling, not by calling its private _send method or mocking control.
        cancelled = threading.Event()
        result = {}

        def read_prompt():
            try:
                result['tokens'] = [t for t in engine.generate_batched(list(range(140, 240)), 8, {}, cancelled) if t is not None]
                result['last'] = dict(engine.last)  # last is thread-local in batch
            except BaseException as error:
                result['error'] = repr(error)

        reader = threading.Thread(target=read_prompt, daemon=True)
        reader.start()
        deadline = time.monotonic() + 5
        while reader.is_alive() and not (engine.progress and engine.progress[1] == 100 and 0 < engine.progress[0] < 100):
            assert time.monotonic() < deadline, ('no in-flight prompt progress', engine.progress, result)
            time.sleep(.002)
        assert reader.is_alive(), result
        cancelled.set()
        reader.join(10)
        assert not reader.is_alive() and 'error' not in result, result
        assert result['tokens'] == [] and result['last']['finish'] == 'cancel', result
        assert 0 < result['last']['prompt_read'] < 100, result
        assert engine.alive() and not any(engine.slot_busy), engine.slot_busy

        # Consumer closes an ordinary stream immediately after an actual T.
        # Its generator must STOP and drain DONE before another request starts.
        stream = engine.generate_batched([30, 31], 40, {}, threading.Event())
        try:
            while True:
                first = next(stream)
                if first is not None:
                    break
            assert first == 32, first
        finally:
            stream.close()
        assert engine.last['finish'] == 'cancel' and engine.alive(), engine.last

        # With another request in flight, disconnect while this consumer is
        # decoding in a batch slot. The original server owns BSTOP/BDONE and
        # its background release; the independent reply must remain complete.
        other = {}
        started = threading.Event()

        def independent():
            try:
                output = []
                for token in engine.generate_batched([1, 2], 60, {}, threading.Event()):
                    if token is not None:
                        output.append(token)
                        started.set()
                other['tokens'] = output
                other['last'] = dict(engine.last)
            except BaseException as error:
                other['error'] = repr(error)
                started.set()

        worker = threading.Thread(target=independent, daemon=True)
        worker.start()
        assert started.wait(5), 'independent request never started'
        stream = engine.generate_batched([90, 91], 40, {}, threading.Event())
        partial = []
        try:
            while len(partial) < 3:
                token = next(stream)
                if token is not None:
                    partial.append(token)
            assert partial == [92, 93, 94], partial
            assert any(s and s.get('state') == 'decoding' for s in engine.slot_live), engine.slot_live
        finally:
            stream.close()
        worker.join(15)
        assert not worker.is_alive() and 'error' not in other, other
        assert other['tokens'] == list(range(3, 63)) and other['last']['finish'] == 'length', other
        deadline = time.monotonic() + 5
        with engine.slot_cv:
            while any(engine.slot_busy):
                assert time.monotonic() < deadline, (engine.slot_busy, engine.slot_releasing)
                engine.slot_cv.wait(.05)
        assert not engine.slot_claims and not engine.slot_releasing, (engine.slot_claims, engine.slot_releasing)
        assert [t for t in engine.generate_batched([120, 121], 2, {}, threading.Event()) if t is not None] == [122, 123]
    finally:
        engine.close()
    print('PASS original server prompt cancellation, ordinary consumer close/drain, batch consumer close/BSTOP/background release, independent full reply and subsequent request; host fixture only')


def bgeni_vision(executable):
    # An empty SVE1 file is a legal GENI request with no pictures (the table is
    # the text continuation). BGENI must take upstream's `GENI 1` rewrite
    # (generate.cpp:10349-10362) and install that request's vision state before
    # any Model work instead of the port's old named refusal.
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / 'empty.sve1'
        path.write_bytes(b'')
        p = Pipe(executable, True, ('--vision',))
        try:
            p.send('BGENI 0 2 temperature=0 ' + str(path) + ' 11,12,13\n')
            p.until('BADM 0 1')
            p.until('BDONE 0 ')
            p.send('QUIT\n')
            lines = p.finish()
            done = [d for d in frames(lines, 'DONE ') if d[2] == '3']
            assert len(done) == 1 and done[0][1] == '1' and done[0][5] == 'length', lines
            assert [f[1] for f in frames(lines, 'T ')] == ['14'], lines
            assert [f[1:3] for f in frames(lines, 'BT ')] == [['0', '15']], lines
            assert [f[1:4] for f in frames(lines, 'BDONE ')] == [['0', '2', 'length']], lines
            assert not frames(lines, 'ERR '), lines
            assert not any('BGENI' in line for line in lines), lines
        finally:
            p.close()


def info_vision_fact(executable):
    # INFO must state the request-admission gate itself: vision=1/0 and, when enabled, the n_embd an SVE1
    # record has to carry. A disabled engine reports 0/0 instead of omitting the keys.
    for batch in (False, True):
        p = Pipe(executable, batch)
        try:
            info = p.seen[0]
            assert info.startswith('INFO backend=hpu'), info
            facts = dict(token.split('=', 1) for token in info.split() if '=' in token)
            assert (facts.get('vision'), facts.get('vision_width')) == ('0', '0'), info
            p.send('QUIT\n')
            p.finish()
        finally:
            p.close()
    p = Pipe(executable, True, ('--vision',))
    try:
        info = p.seen[0]
        facts = dict(token.split('=', 1) for token in info.split() if '=' in token)
        assert (facts.get('vision'), facts.get('vision_width')) == ('1', '16'), info
        p.send('QUIT\n')
        p.finish()
    finally:
        p.close()


if __name__ == '__main__':
    executable = str(pathlib.Path(sys.argv[1]).resolve())
    fifo_quit_eof(executable)
    eof_while_active(executable)
    ordinary_stop(executable)
    bgen_stop_resume(executable)
    stop_bstop_scopes(executable)
    invalid_then_valid(executable)
    bgeni_vision(executable)
    info_vision_fact(executable)
    print('PASS FIFO/QUIT/EOF, ordinary STOP scope, BGEN in-window committed cancellation/resume, idle STOP, active/free BSTOP, BGENI vision admission, INFO vision gate and strict terminal/read counts; production C++ host fixture, no device')
    if '--server-adapter' in sys.argv:
        server_adapter(executable)
        server_cleanup(executable)
