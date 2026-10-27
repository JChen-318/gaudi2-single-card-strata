"""Regression tests for no-device validation gates; imports never rewrite bound records."""
import contextlib
import importlib.util
import io
import pathlib
import subprocess
import tempfile
from unittest.mock import patch

ROOT=pathlib.Path(__file__).resolve().parents[2]

def load(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def main():
    reader=load('reader_gate','work/ple_reader_check.py')
    values=load('value_gate','work/audit_cli_values.py')
    formats='ple formats, both readers vs their dequantizers: IQ4_NL: OK\n'
    def output(first='OK',second='OK',extra=''):
        return formats+f'ple_reader selftest (row_bytes=90): {first}\nple_reader selftest (row_bytes=110): {second}\n'+extra
    cases=[
        (0,output(),True),
        (9,output(),False),
        (-11,output(),False),
        (1,output('OK','FAILED'),False),
        (0,output('OK','FAILED'),False),
        (1,output('FAILED','FAILED','FAIL src/ngram/ple_reader_test.cpp:343: keep-alive: 4 reads in 300 ms at a 20 ms period\n'),True),
        (1,output('FAILED','FAILED','FAIL src/ngram/ple_reader_test.cpp:343: keep-alive: 0 reads in 300 ms at a 20 ms period\n'),False),
        (1,output('FAILED','FAILED','FAIL src/ngram/ple_reader_test.cpp:355: the keep-alive did not start again (4 -> 6)\n'),True),
        (1,output('FAILED','FAILED','FAIL src/ngram/ple_reader_test.cpp:355: the keep-alive did not start again (4 -> 4)\n'),False),
        (1,output('FAILED','FAILED','FAIL src/ngram/ple_reader_test.cpp:326: no read was held back\n'),True),
        (1,output('FAILED','FAILED','FAIL src/ngram/ple_reader_test.cpp:326: no read was held back\nFAIL src/ngram/ple_reader_test.cpp:325: injected delay not observed (10 us)\n'),False),
        (1,output('FAILED','FAILED','FAIL src/ngram/ple_reader_test.cpp:343: keep-alive: 4 reads in 300 ms at a 20 ms period\nERROR: AddressSanitizer: invalid free\n'),False),
        (9,output('FAILED','FAILED','FAIL src/ngram/ple_reader_test.cpp:343: keep-alive: 4 reads in 300 ms at a 20 ms period\n'),False),
        (1,output('FAILED','FAILED','FAIL different.cpp:343: keep-alive: 4 reads in 300 ms at a 20 ms period\n'),False),
        (0,output().replace('row_bytes=110','row_bytes=96'),False),
        (0,output()+output(),False),
        (0,output().replace(formats,''),False),
    ]
    for returncode,text,expected in cases:
        assert reader.evaluate(returncode,text)['ok']==expected,(returncode,text)
    # Exercise the actual wrapper entry, not just its new pure evaluator.
    with tempfile.TemporaryDirectory(prefix='strata-validation-gates-') as directory:
        for index,(returncode,text,expected) in enumerate(cases):
            record=pathlib.Path(directory)/f'{index}.json'
            result=subprocess.CompletedProcess([],returncode,text,'')
            with patch.object(reader.subprocess,'run',return_value=result),contextlib.redirect_stdout(io.StringIO()):
                rc=reader.main(['reader-check','fixture','fixture-dir',str(record)])
            assert (rc==0)==expected and record.is_file(),(index,rc,expected)
    # Bounds require the operand too, and code rather than comments/strings.
    assert not values.comparison_present('number<uint32_t>(value())','< 1')
    assert not values.comparison_present('number<uint32_t>(value())','> 1024')
    assert not values.comparison_present('if(x > 999) {} // > 1024','> 1024')
    assert not values.comparison_present('error("> 1024");','> 1024')
    assert not values.comparison_present('if(x <= 1) throw error;','< 1')
    assert values.comparison_present('if(x < 1 || x > 1024) throw error;','> 1024')
    actual=values.audit_text(values.UPSTREAM,values.PORT);assert actual['ported_options']>=80
    for flag,resolver in values.REQUIRED_CALLS.items():
        # The named checked call cannot disappear behind a curated verdict.
        import re
        branch=values.port_branch(flag,values.PORT)
        changed=re.sub(r'\b'+resolver+r'\s*\(', 'unchecked_replacement(',branch,count=1)
        assert changed!=branch,flag
        mutated=values.PORT.replace(branch,changed,1)
        refused=False
        with contextlib.redirect_stdout(io.StringIO()):
            try: values.audit_text(values.UPSTREAM,mutated)
            except ValueError as error: refused='required checked resolver call missing' in str(error)
        assert refused,flag
    print(f'PASS validation gates: {len(cases)} reader exit/summary/semantic/fatal cases through evaluator+main; template/comment/string bounds and five removed checked-call controls; no device')

if __name__=='__main__':
    main()
