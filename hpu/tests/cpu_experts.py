import pathlib
import struct
import subprocess
import sys
import tempfile
from model_plan import fixture


def main(executable, nv=False):
    with tempfile.TemporaryDirectory(prefix='strata-hpu-cpu-experts-') as directory:
        path = pathlib.Path(directory) / 'model.gguf'
        offsets = fixture(path,nvfp4_roles=('gate','up','down') if nv else ())
        with path.open('r+b') as stream:
            for role, (name, rows, columns) in enumerate([('gate', 640, 2560), ('up', 640, 2560), ('down', 2560, 640)]):
                stream.seek(offsets[f'blk.0.ffn_{name}_exps.weight'])
                for row in range(rows):
                    code = (row + role + 2) % 4
                    packed = code | code << 2 | code << 4 | code << 6
                    block = bytes([0x20 + role * 2]) * 4 + bytes([((row+role)%16) | ((row+3)%16)<<4]) * 32 if nv else struct.pack('<H', 0x2400) + bytes([packed]) * 16
                    stream.write(block * (columns // 64))
                if nv:
                    stream.seek(offsets[f'blk.0.ffn_{name}_exps.scale'])
                    factor=[0.125,0.75,1/32768][role]
                    stream.write(struct.pack('<512f', *[factor*(i%7+1) for i in range(512)]))
        subprocess.run([executable, str(path)], check=True)
        from native_pack_model import fixture as pack_fixture
        pack = pathlib.Path(directory) / 'native-pack'
        pack_fixture(pack, path, binary=True, populated=[(0,0)])
        subprocess.run([executable, str(path), str(pack)], check=True)
        if not nv:
            (pack / 'experts.bin').unlink()
            subprocess.run([executable, str(path), str(pack)], check=True)


if __name__ == '__main__':
    main(sys.argv[1], '--nvfp4' in sys.argv[2:])
