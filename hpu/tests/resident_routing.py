"""Real production protocol/Generation/SlotSession, host fixture only; no HPU."""
import pathlib
import struct
import sys
import tempfile

from request_control import Pipe, frames


def records(path):
    data = path.read_bytes()
    assert len(data) % 24 == 0, ('partial resident routing record', len(data))
    for offset in range(0, len(data), 24):
        layer, k, a, b, x, y = struct.unpack_from('<4i2f', data, offset)
        assert layer in (0, 1) and k == 2 and 0 <= a < 16 and 0 <= b < 16
        assert (x, y) == (.25, .75)
    return data


def check(executable, batch, directory):
    path = pathlib.Path(directory) / ('batch.bin' if batch else 'ordinary.bin')
    # A new process truncates ONCE, even when the prior process left a trace.
    path.write_bytes(b'old process payload')
    p = Pipe(executable, batch, extra=['--serve', '--dump-routing', str(path)])
    try:
        assert path.read_bytes() == b''
        p.send('GEN 3 temperature=0 1,2\n')
        p.until('DONE ')
        assert [int(f[1]) for f in frames(p.seen, 'T ')] == [3, 4, 5]
        first = records(path)
        assert len(first) == 8 * 24, ('missing scalar prompt/decode routing', len(first))
        p.send('GEN 1 temperature=0 1,2\n')
        done = p.until('DONE ').split()
        assert int(done[14]) == 0 and records(path) == first, ('fully cached trace repeated or truncated', done)
        if batch:
            p.send('BGEN 0 5 temperature=0 20,21\n')
            p.until('DONE ')
            p.until('BADM 0 1')
            p.until('BDONE 0 ')
            assert [int(f[2]) for f in frames(p.seen, 'BT 0 ')] == [23, 24, 25, 26]
            after = records(path)
            assert len(after) == len(first) + 12 * 24 and after.startswith(first), ('slot prompt or continuation lost', len(after))
        else:
            p.send('GEN 2 temperature=0 7,8\n')
            p.until('DONE ')
            after = records(path)
            assert len(after) == len(first) + 6 * 24 and after.startswith(first)
        # A rejected request must not reopen/truncate the global file.
        p.send('GEN 1 invalid_key=1 30,31\n')
        p.until('ERR ', errors=True)
        assert records(path) == after
        p.send('GEN 1 temperature=0 40,41\n')
        p.until('DONE ')
        final = records(path)
        assert final.startswith(after) and len(final) > len(after)
        p.send('QUIT\n')
        p.finish()
        assert records(path) == final
    finally:
        p.close()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='strata-resident-routing-') as directory:
        for batch in (False, True):
            check(sys.argv[1], batch, directory)
    print('PASS resident routing real pipes: process-open once, GEN/cache/BGEN/BT/BDONE/error/recovery/QUIT append and checked record bytes; host fixture only')
