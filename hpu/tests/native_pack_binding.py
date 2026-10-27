"""Real main-GGUF/pack binding without native CPU workers or SDK initialization."""
import pathlib
import subprocess
import sys
import tempfile
from model_plan import fixture, allocated_bytes
from native_pack_model import fixture as pack_fixture

with tempfile.TemporaryDirectory(prefix='strata-hpu-pack-binding-') as directory:
    root=pathlib.Path(directory)
    main=root/'main.gguf'
    pack=root/'pack'
    fixture(main)
    pack_fixture(pack,main)
    assert allocated_bytes(main)<64<<20
    subprocess.run([sys.argv[1],str(main),str(pack)],check=True)
    pack_fixture(pack,main,binary=True)
    assert allocated_bytes(pack/'experts.bin')<64<<20
    subprocess.run([sys.argv[1],str(main),str(pack)],check=True)
    for roles in [('gate','up','down'),('down',)]:
        (pack/'experts.bin').unlink()
        (pack/'experts.bin.src.json').unlink()
        fixture(main,nvfp4_roles=roles)
        pack_fixture(pack,main,binary=True)
        assert allocated_bytes(main)<64<<20 and allocated_bytes(pack/'experts.bin')<64<<20
        print(f'sparse fixture logical_bytes={(pack/"experts.bin").stat().st_size} allocated_bytes={allocated_bytes(pack/"experts.bin")}',flush=True)
        subprocess.run([sys.argv[1],str(main),str(pack)],check=True)
print('PASS full actual GGUF metadata/native source binding, original layout/provenance producer and sparse payloads; no device')
