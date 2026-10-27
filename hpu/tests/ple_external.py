"""Actual PLE input with model-owned hashes and a separate sparse SSD table."""
import pathlib
import subprocess
import sys
import tempfile
from model_plan import fixture, external_ple_fixture

with tempfile.TemporaryDirectory(prefix='strata-hpu-ple-external-') as directory:
    root = pathlib.Path(directory)
    main = root / 'main.gguf'
    external = root / 'table-00002-of-00002.gguf'
    fixture(main, missing='per_layer_token_embd.weight')
    external_ple_fixture(external)
    subprocess.run([sys.argv[1], str(main), str(external)], check=True)
print('PASS external PLE mmap/Direct bounded gather, main-owned hash constants and chronological history; no device')
